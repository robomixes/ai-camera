"""Find an RTSP camera that has moved to a new address on its own subnet.

A camera that reboots can come back on a different DHCP lease. The capture
backend will retry the old address forever, so the system stays blind until
someone notices and edits the config by hand. This module locates the camera
again and proves it is the right one before anything is changed.

SECURITY: verification connects with the camera's real credentials, so those
credentials reach whichever host is being verified. Candidates are gated on an
unauthenticated TCP probe first, which on a home subnet narrows 254 addresses
to a handful, but the exposure is not zero: a hostile device listening on the
RTSP port could harvest them. That is inherent to finding a camera by address
rather than by hardware identity. It is why discovery can be disabled and why
the scan range can be pinned to an explicit CIDR.

Never log a URL from this module. Log the host and the outcome.
"""

import ipaddress
import logging
import os
import socket
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_RTSP_PORT = 554

# cv2.VideoCapture reads OPENCV_FFMPEG_CAPTURE_OPTIONS from the process
# environment when it opens. Two threads opening captures at once therefore
# race over a single global, and the loser silently gets the other's transport
# and timeouts. Every open in this project must hold this lock across the
# set-then-open pair.
_capture_options_lock = threading.Lock()


def open_capture_locked(url: str, transport: str, timeout_seconds: float):
    """Open a cv2 capture with FFmpeg options applied safely.

    Returns an opened VideoCapture, or None. The caller owns releasing it.
    """
    import cv2  # local: keeps this module importable without cv2 for unit tests

    micros = max(int(timeout_seconds * 1_000_000), 1_000_000)
    with _capture_options_lock:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}"
            f"|stimeout;{micros}"
            f"|timeout;{micros}"
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    return cap


# --- URL handling -----------------------------------------------------------

@dataclass(frozen=True)
class RtspUrlParts:
    """A parsed RTSP URL. `userinfo` is opaque and must never be logged."""
    scheme: str
    userinfo: str | None
    host: str
    port: int | None
    rest: str          # path + params + query, verbatim


def parse_rtsp_url(url: str) -> RtspUrlParts | None:
    """Split an RTSP URL without decomposing the credentials.

    urllib.parse is deliberately not used here. RTSP passwords in the wild
    contain '@', ':' and '/' unencoded, and urlsplit puts the wrong text in
    .hostname for those. Splitting the authority on its *last* '@' is correct
    even when the password contains one.
    """
    if not url or "://" not in url:
        return None
    scheme, _, remainder = url.partition("://")
    if not scheme or not remainder:
        return None

    # Split on the LAST '@' in the whole remainder, before looking for the path
    # separator. Cutting the authority at the first '/' instead would land
    # inside any password containing '/', mis-parsing the host and — far worse
    # — leaking that password through mask_url().
    #
    # The cost is that a path containing '@' would be misread as credentials.
    # RTSP stream paths do not contain '@' in practice, whereas passwords
    # containing '/' are entirely ordinary, so this is the safer trade.
    if "@" in remainder:
        userinfo, _, hostpart = remainder.rpartition("@")
        if not userinfo:
            userinfo = None
    else:
        userinfo, hostpart = None, remainder

    slash = hostpart.find("/")
    if slash == -1:
        hostport, rest = hostpart, ""
    else:
        hostport, rest = hostpart[:slash], hostpart[slash:]
    if not hostport:
        return None

    # Bracketed IPv6 literal, e.g. [2001:db8::1]:554
    if hostport.startswith("["):
        close = hostport.find("]")
        if close == -1:
            return None
        host = hostport[: close + 1]
        tail = hostport[close + 1 :]
        port_str = tail[1:] if tail.startswith(":") else ""
    elif hostport.count(":") == 1:
        host, _, port_str = hostport.partition(":")
    else:
        host, port_str = hostport, ""

    if not host:
        return None
    port = None
    if port_str:
        try:
            port = int(port_str)
        except ValueError:
            return None

    return RtspUrlParts(scheme=scheme, userinfo=userinfo, host=host, port=port, rest=rest)


def rebuild_rtsp_url(parts: RtspUrlParts, host: str, port: int | None = None) -> str:
    """Rebuild a URL with a different host, changing nothing else."""
    use_port = parts.port if port is None else port
    hostport = f"{host}:{use_port}" if use_port is not None else host
    authority = f"{parts.userinfo}@{hostport}" if parts.userinfo else hostport
    return f"{parts.scheme}://{authority}{parts.rest}"


def rtsp_port(parts: RtspUrlParts) -> int:
    return parts.port or DEFAULT_RTSP_PORT


def mask_url(url: str) -> str:
    """Hide the password in a URL so it can be logged or returned by the API.

    Handles the cases the previous implementation got wrong: a password
    containing '@' (it split on the first one, emitting the password's tail in
    the host position) and a URL with a username but no password (it emitted a
    mangled 'rtsp:****@host').
    """
    parts = parse_rtsp_url(url)
    if parts is None:
        return url
    if not parts.userinfo:
        return url
    user, sep, _password = parts.userinfo.partition(":")
    if not sep:
        return url                       # username only, nothing secret to hide
    hostport = f"{parts.host}:{parts.port}" if parts.port is not None else parts.host
    return f"{parts.scheme}://{user}:****@{hostport}{parts.rest}"


def mask_host_only(url: str) -> str:
    """Reduce a URL to scheme and host. Safest form for a log line."""
    parts = parse_rtsp_url(url)
    if parts is None:
        return "<unparseable url>"
    return f"{parts.scheme}://{parts.host}"


# --- Subnet derivation ------------------------------------------------------

def candidate_hosts(
    current_host: str,
    prefix_len: int = 24,
    explicit_cidr: str = "",
    max_hosts: int = 512,
    exclude: set[str] | None = None,
) -> list[str]:
    """Addresses to probe, nearest to the camera's last known address first.

    The range is derived from the camera's own configured host, never from a
    local interface: inside Docker the container sits on a bridge network
    (172.x) while the camera is on the LAN, so local interfaces would send the
    scan to entirely the wrong place.
    """
    exclude = exclude or set()

    network = None
    anchor = None

    if explicit_cidr:
        try:
            network = ipaddress.ip_network(explicit_cidr, strict=False)
        except ValueError:
            logger.warning("Ignoring invalid discovery subnet %r", explicit_cidr)
            network = None

    if network is None:
        resolved = current_host
        try:
            ipaddress.ip_address(current_host)
        except ValueError:
            # A hostname: if DNS resolves it, that is already the answer.
            try:
                resolved = socket.gethostbyname(current_host)
            except OSError:
                logger.warning(
                    "Cannot derive a scan range from hostname %r; "
                    "set an explicit discovery subnet", current_host
                )
                return []
        try:
            addr = ipaddress.ip_address(resolved)
        except ValueError:
            return []
        if addr.version != 4:
            logger.warning("Discovery supports IPv4 only; skipping %s", current_host)
            return []
        anchor = addr
        try:
            network = ipaddress.ip_network(f"{resolved}/{prefix_len}", strict=False)
        except ValueError:
            return []

    if network.version != 4:
        return []
    if network.num_addresses - 2 > max_hosts:
        logger.warning(
            "Scan range %s has %d hosts, above the %d limit; refusing to scan",
            network, network.num_addresses - 2, max_hosts,
        )
        return []

    hosts = [str(h) for h in network.hosts() if str(h) not in exclude]
    if anchor is not None:
        # DHCP tends to hand out adjacent leases, and a reboot often gets the
        # same one back, so probe outward from where the camera used to be.
        hosts.sort(key=lambda h: abs(int(ipaddress.ip_address(h)) - int(anchor)))
    return hosts


# --- Probing and verification ----------------------------------------------

def probe_port(host: str, port: int = DEFAULT_RTSP_PORT, timeout: float = 0.5) -> bool:
    """Is anything listening? A bare TCP connect; no data and no credentials."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def scan_ports(
    hosts: list[str],
    port: int = DEFAULT_RTSP_PORT,
    timeout: float = 0.5,
    concurrency: int = 32,
) -> list[str]:
    """Hosts with the port open, in the order given."""
    from concurrent.futures import ThreadPoolExecutor

    if not hosts:
        return []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(lambda h: probe_port(h, port, timeout), hosts))
    return [h for h, open_ in zip(hosts, results) if open_]


@dataclass
class VerifyResult:
    host: str
    ok: bool = False
    width: int = 0
    height: int = 0
    frames: int = 0
    scene_diff: float | None = None
    reason: str = ""


def _default_open(url: str, transport: str, timeout: float):
    return open_capture_locked(url, transport, timeout)


def verify_candidate(
    parts: RtspUrlParts,
    host: str,
    transport: str = "tcp",
    expect_size: tuple[int, int] | None = None,
    expect_signature=None,
    *,
    require_resolution_match: bool = True,
    scene_max_diff: float | None = None,
    min_frames: int = 2,
    timeout: float = 6.0,
    open_capture=_default_open,
    signature_fn=None,
) -> VerifyResult:
    """Decide whether the stream at `host` is the camera described by `parts`.

    A decoded frame alone proves almost nothing — a neighbour's camera of the
    same make, reached with reused credentials, decodes a frame too, and
    adopting it would quietly start recording someone else's property. So the
    checks layer up: same stream path, several consecutive frames, matching
    resolution, and finally the scene itself. A camera that merely rebooted is
    still pointing at the same view; a different camera is not.
    """
    result = VerifyResult(host=host)
    url = rebuild_rtsp_url(parts, host)

    cap = None
    try:
        cap = open_capture(url, transport, timeout)
        if cap is None:
            result.reason = "open_failed"
            return result

        frame = None
        for _ in range(min_frames):
            ok, f = cap.read()
            if not ok or f is None:
                result.reason = "no_frames"
                return result
            result.frames += 1
            frame = f

        result.height, result.width = frame.shape[:2]

        if require_resolution_match and expect_size:
            if (result.width, result.height) != tuple(expect_size):
                result.reason = "resolution_mismatch"
                return result

        if expect_signature is not None and signature_fn is not None and scene_max_diff is not None:
            candidate_sig = signature_fn(frame)
            if candidate_sig is None:
                result.reason = "no_signature"
                return result
            import numpy as np
            diff = float(np.mean(np.abs(expect_signature.astype(np.int16)
                                        - candidate_sig.astype(np.int16))))
            result.scene_diff = diff
            if diff > scene_max_diff:
                result.reason = "scene_mismatch"
                return result

        result.ok = True
        result.reason = "ok"
        return result
    except Exception as e:
        result.reason = f"error:{type(e).__name__}"
        return result
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


# --- Multi-camera orchestration --------------------------------------------

@dataclass
class DiscoveryTarget:
    camera_id: str
    url: str                                  # real, never logged
    transport: str = "tcp"
    expect_size: tuple[int, int] | None = None
    expect_signature=None


@dataclass
class DiscoveryOutcome:
    camera_id: str
    status: str = "not_found"                 # found | not_found | ambiguous | skipped
    new_url: str = ""                         # real, never logged
    new_host: str = ""
    candidates: list[str] = field(default_factory=list)
    detail: str = ""


def rediscover(
    targets: list[DiscoveryTarget],
    excluded_hosts: set[str] | None = None,
    *,
    prefix_len: int = 24,
    explicit_cidr: str = "",
    max_hosts: int = 512,
    port_timeout: float = 0.5,
    port_concurrency: int = 32,
    verify_timeout: float = 6.0,
    min_frames: int = 2,
    require_resolution_match: bool = True,
    scene_max_diff: float | None = None,
    open_capture=_default_open,
    signature_fn=None,
) -> list[DiscoveryOutcome]:
    """Locate every offline camera in one pass.

    One port sweep serves all cameras on the same network, and a host adopted
    by one camera is withdrawn from every other camera's candidates so two
    cameras can never be pointed at the same stream.
    """
    excluded_hosts = set(excluded_hosts or set())
    outcomes: list[DiscoveryOutcome] = []
    assigned: dict[str, str] = {}
    sweep_cache: dict[tuple[str, int], list[str]] = {}

    for target in sorted(targets, key=lambda t: t.camera_id):
        outcome = DiscoveryOutcome(camera_id=target.camera_id)
        parts = parse_rtsp_url(target.url)
        if parts is None:
            outcome.status = "skipped"
            outcome.detail = "url not parseable"
            outcomes.append(outcome)
            continue

        port = rtsp_port(parts)
        hosts = candidate_hosts(
            parts.host, prefix_len=prefix_len, explicit_cidr=explicit_cidr,
            max_hosts=max_hosts, exclude=excluded_hosts | set(assigned),
        )
        if not hosts:
            outcome.status = "skipped"
            outcome.detail = "no scan range"
            outcomes.append(outcome)
            continue

        cache_key = (hosts[0].rsplit(".", 1)[0], port)
        if cache_key in sweep_cache:
            open_hosts = [h for h in sweep_cache[cache_key] if h not in assigned]
        else:
            open_hosts = scan_ports(hosts, port, port_timeout, port_concurrency)
            sweep_cache[cache_key] = list(open_hosts)
        logger.info(
            "[%s] scanned %d host(s), %d with port %d open",
            target.camera_id, len(hosts), len(open_hosts), port,
        )

        matches: list[VerifyResult] = []
        for host in open_hosts:
            res = verify_candidate(
                parts, host, target.transport, target.expect_size,
                target.expect_signature,
                require_resolution_match=require_resolution_match,
                scene_max_diff=scene_max_diff, min_frames=min_frames,
                timeout=verify_timeout, open_capture=open_capture,
                signature_fn=signature_fn,
            )
            logger.info("[%s] candidate %s: %s", target.camera_id, host, res.reason)
            if res.ok:
                matches.append(res)

        if len(matches) == 1:
            host = matches[0].host
            assigned[host] = target.camera_id
            outcome.status = "found"
            outcome.new_host = host
            outcome.new_url = rebuild_rtsp_url(parts, host)
        elif len(matches) > 1:
            # Two devices we cannot tell apart. Guessing risks attaching this
            # camera's events, recordings and recognised faces to the wrong
            # view, which is both severe and hard to notice.
            outcome.status = "ambiguous"
            outcome.candidates = [m.host for m in matches]
            outcome.detail = "more than one device matched"
            logger.warning(
                "[%s] %d devices matched (%s) — adopting none",
                target.camera_id, len(matches), ", ".join(outcome.candidates),
            )
        else:
            outcome.detail = "no matching stream found"

        outcomes.append(outcome)

    return outcomes
