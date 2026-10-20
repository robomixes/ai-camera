"""Decide when to go looking for a camera that has stopped responding.

The capture backend retries its configured address forever, so a camera that
came back on a different DHCP lease stays lost until a human notices. This
service watches camera health, waits long enough to be sure the camera is
really gone rather than merely rebooting, then asks camera.discovery to find
it — and hands the result to a callback that persists and reattaches it.

Nothing here blocks the event loop: scanning and reattaching both run in a
worker thread.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import config
from camera import discovery

logger = logging.getLogger(__name__)


def _cfg(name: str, default):
    """Read a setting at call time so dashboard changes apply without a restart."""
    return getattr(config, name, default)


@dataclass
class _CamState:
    first_unhealthy_at: float | None = None
    next_scan_at: float = 0.0
    backoff: float = 0.0
    scans: int = 0
    status: str = "ok"            # ok | waiting | searching | ambiguous | lost
    detail: str = ""
    candidates: list[str] = field(default_factory=list)
    last_change_at: float = 0.0
    last_good_signature: object = None
    last_good_size: tuple[int, int] | None = None


class CameraDiscoveryService:
    def __init__(self, manager, persist_cb):
        self._manager = manager
        self._persist = persist_cb
        self._state: dict[str, _CamState] = {}
        self._scanning = False     # only touched from the event loop

    # --- health -----------------------------------------------------------

    def _instances(self):
        try:
            return list(self._manager._instances.items())
        except Exception:
            return []

    def _is_healthy(self, cam) -> bool:
        if cam is None:
            return False
        try:
            if hasattr(cam, "has_signal"):
                return bool(cam.has_signal())
            return bool(cam.is_running() and cam.get_frame() is not None)
        except Exception:
            return False

    def _remember_good_frame(self, st: _CamState, cam) -> None:
        """Keep a signature of the live view, to recognise the camera later.

        This is what separates 'our camera moved' from 'some other camera on
        the network answers the same credentials'. It can only be captured
        while the camera is healthy, so it is refreshed on every healthy tick.
        """
        if not _cfg("AUTO_DISCOVERY_REQUIRE_SCENE_MATCH", True):
            return
        try:
            frame = cam.get_frame()
            if frame is None:
                return
            from web.ai_runner import AIRunner
            sig = AIRunner._frame_signature(frame)
            if sig is not None:
                st.last_good_signature = sig
                st.last_good_size = (frame.shape[1], frame.shape[0])
        except Exception:
            pass

    # --- the tick ---------------------------------------------------------

    async def run_once(self, force_ids=()) -> dict:
        now = time.time()
        force = set(force_ids)
        eligible = []

        for cam_id, inst in self._instances():
            st = self._state.setdefault(cam_id, _CamState())
            cam = inst.cam
            cfg = inst.config or {}

            if self._is_healthy(cam):
                if st.status != "ok":
                    logger.info("[%s] camera healthy again", cam_id)
                st.first_unhealthy_at = None
                st.backoff = 0.0
                st.next_scan_at = 0.0
                st.scans = 0
                st.status = "ok"
                st.detail = ""
                st.candidates = []
                self._remember_good_frame(st, cam)
                continue

            # Only an RTSP camera has an address to go looking for.
            if (cfg.get("type") or "rtsp") != "rtsp":
                st.status = "lost"
                st.detail = "not an IP camera"
                continue

            if st.first_unhealthy_at is None:
                st.first_unhealthy_at = now
                st.last_change_at = now
                st.status = "waiting"

            grace = self._grace_for(cfg)
            offline_for = now - st.first_unhealthy_at
            if cam_id in force:
                pass
            elif offline_for < grace:
                st.status = "waiting"
                st.detail = f"offline {int(offline_for)}s, scanning at {int(grace)}s"
                continue
            elif now < st.next_scan_at:
                st.status = "waiting"
                st.detail = f"next search in {int(st.next_scan_at - now)}s"
                continue

            eligible.append((cam_id, inst))

        if not eligible:
            return {"scanned": False, "reason": "nothing eligible"}
        if self._scanning:
            return {"scanned": False, "reason": "a search is already running"}
        if not _cfg("AUTO_DISCOVERY_ENABLED", True):
            return {"scanned": False, "reason": "disabled"}

        # Yield to a user-initiated reconfigure rather than competing with it.
        #
        # This only *tests* the lock and releases it again. Holding it across
        # the scan would deadlock: the apply step runs in a worker thread via
        # asyncio.to_thread and takes the same lock, and an RLock is reentrant
        # only for the thread that already owns it — so a lock taken here, on
        # the event loop, can never be re-entered there. The apply path takes
        # it properly in its own thread instead.
        lock = getattr(self._manager, "reconfigure_lock", None)
        if lock is not None:
            if not lock.acquire(blocking=False):
                return {"scanned": False, "reason": "config change in progress"}
            lock.release()
        return await self._scan(eligible, now)

    def _grace_for(self, cfg: dict) -> float:
        """How long to wait before searching.

        A camera that is merely rebooting is frameless for ~40s (the 30s
        reconnect ceiling plus the socket timeout) and a cold boot can take a
        minute, so the default grace is deliberately long. But if the old
        address actively refuses the RTSP port, the camera is gone rather than
        booting, and there is no reason to keep waiting.
        """
        slow = float(_cfg("AUTO_DISCOVERY_OFFLINE_GRACE_SECONDS", 120))
        fast = float(_cfg("AUTO_DISCOVERY_FAST_GRACE_SECONDS", 30))
        parts = discovery.parse_rtsp_url(cfg.get("url") or "")
        if parts is None:
            return slow
        try:
            if not discovery.probe_port(parts.host, discovery.rtsp_port(parts), 0.4):
                return fast
        except Exception:
            pass
        return slow

    async def _scan(self, eligible, now: float) -> dict:
        self._scanning = True
        try:
            targets = []
            excluded = set()
            # Addresses belonging to cameras that are currently fine must never
            # be offered as candidates for a camera that is not.
            for cam_id, inst in self._instances():
                st = self._state.get(cam_id)
                if st and st.status == "ok":
                    p = discovery.parse_rtsp_url((inst.config or {}).get("url") or "")
                    if p:
                        excluded.add(p.host)

            for cam_id, inst in eligible:
                st = self._state[cam_id]
                st.status = "searching"
                st.detail = "searching the network"
                cfg = inst.config or {}
                # Prefer the size actually measured while the camera was alive.
                # Falling back to the configured dimensions matters because a
                # camera that has never been healthy has no remembered size or
                # scene, and verification would otherwise be reduced to "this
                # host decodes two frames" — which any RTSP device satisfies.
                expect_size = st.last_good_size
                if expect_size is None and cfg.get("width") and cfg.get("height"):
                    expect_size = (int(cfg["width"]), int(cfg["height"]))
                targets.append(discovery.DiscoveryTarget(
                    camera_id=cam_id,
                    url=cfg.get("url") or "",
                    transport=cfg.get("transport") or "tcp",
                    expect_size=expect_size,
                    expect_signature=st.last_good_signature,
                ))

            sig_fn = None
            if _cfg("AUTO_DISCOVERY_REQUIRE_SCENE_MATCH", True):
                from web.ai_runner import AIRunner
                sig_fn = AIRunner._frame_signature

            outcomes = await asyncio.wait_for(asyncio.to_thread(
                discovery.rediscover,
                targets, excluded,
                prefix_len=int(_cfg("AUTO_DISCOVERY_SUBNET_PREFIX", 24)),
                explicit_cidr=str(_cfg("AUTO_DISCOVERY_SUBNET", "") or ""),
                max_hosts=int(_cfg("AUTO_DISCOVERY_MAX_HOSTS", 512)),
                port_timeout=float(_cfg("AUTO_DISCOVERY_PORT_TIMEOUT", 0.5)),
                port_concurrency=int(_cfg("AUTO_DISCOVERY_PORT_CONCURRENCY", 32)),
                verify_timeout=float(_cfg("AUTO_DISCOVERY_VERIFY_TIMEOUT_SECONDS", 6.0)),
                min_frames=int(_cfg("AUTO_DISCOVERY_MIN_FRAMES", 2)),
                require_resolution_match=bool(_cfg("AUTO_DISCOVERY_REQUIRE_RESOLUTION_MATCH", True)),
                scene_max_diff=(float(_cfg("AUTO_DISCOVERY_SCENE_MAX_DIFF", 45.0))
                                if sig_fn is not None else None),
                signature_fn=sig_fn,
            ), timeout=float(_cfg("AUTO_DISCOVERY_SCAN_TIMEOUT_SECONDS", 180)))

            applied = []
            for outcome in outcomes:
                st = self._state[outcome.camera_id]
                st.scans += 1
                st.last_change_at = time.time()

                if outcome.status == "found":
                    if not _cfg("AUTO_DISCOVERY_AUTO_APPLY", True):
                        st.status = "ambiguous"
                        st.candidates = [outcome.new_host]
                        st.detail = f"found at {outcome.new_host} — apply it to reattach"
                        logger.info("[%s] found at %s (auto-apply is off)",
                                    outcome.camera_id, outcome.new_host)
                    else:
                        ok = await asyncio.to_thread(
                            self._persist, outcome.camera_id, outcome.new_url)
                        if ok:
                            st.status = "ok"
                            st.detail = f"reattached at {outcome.new_host}"
                            st.first_unhealthy_at = None
                            st.backoff = 0.0
                            st.next_scan_at = 0.0
                            applied.append((outcome.camera_id, outcome.new_host))
                            logger.info("[%s] reattached at %s", outcome.camera_id, outcome.new_host)
                        else:
                            st.status = "lost"
                            st.detail = "found it but could not apply the new address"
                            self._back_off(st)
                elif outcome.status == "ambiguous":
                    # Deliberately adopt nothing: attaching to the wrong device
                    # would quietly record someone else's property.
                    st.status = "ambiguous"
                    st.candidates = outcome.candidates
                    st.detail = f"{len(outcome.candidates)} devices matched — choose one"
                    self._back_off(st)
                else:
                    st.status = "lost"
                    st.detail = outcome.detail or "not found"
                    self._back_off(st)

            return {"scanned": True, "applied": applied,
                    "outcomes": [o.camera_id + ":" + o.status for o in outcomes]}
        except asyncio.TimeoutError:
            for cam_id, _ in eligible:
                st = self._state[cam_id]
                st.status = "lost"
                st.detail = "search timed out"
                self._back_off(st)
            logger.error("Rediscovery scan timed out")
            return {"scanned": False, "reason": "timeout"}
        finally:
            self._scanning = False

    def _back_off(self, st: _CamState) -> None:
        """Search soon at first, then progressively less often — but never stop.

        A hard stop after N attempts would switch the feature off during
        exactly the long outage it exists for.
        """
        lo = float(_cfg("AUTO_DISCOVERY_MIN_INTERVAL_SECONDS", 300))
        hi = float(_cfg("AUTO_DISCOVERY_MAX_INTERVAL_SECONDS", 1800))
        st.backoff = min(max(st.backoff * 2, lo), hi)
        st.next_scan_at = time.time() + st.backoff

    # --- API surface ------------------------------------------------------

    def request_scan(self, camera_id: str) -> bool:
        st = self._state.get(camera_id)
        if st is None:
            return False
        st.next_scan_at = 0.0
        st.backoff = 0.0
        if st.first_unhealthy_at is None:
            st.first_unhealthy_at = time.time()
        return True

    async def adopt(self, camera_id: str, host: str) -> bool:
        """Apply a candidate the user picked from an ambiguous result."""
        st = self._state.get(camera_id)
        if st is None or host not in st.candidates:
            return False
        inst = dict(self._instances()).get(camera_id)
        if inst is None:
            return False
        parts = discovery.parse_rtsp_url((inst.config or {}).get("url") or "")
        if parts is None:
            return False
        new_url = discovery.rebuild_rtsp_url(parts, host)
        ok = await asyncio.to_thread(self._persist, camera_id, new_url)
        if ok:
            st.status = "ok"
            st.detail = f"reattached at {host}"
            st.candidates = []
            st.first_unhealthy_at = None
            st.backoff = 0.0
            st.next_scan_at = 0.0
            logger.info("[%s] reattached at %s (chosen by user)", camera_id, host)
        return ok

    def status(self) -> dict:
        now = time.time()
        out = {}
        for cam_id, st in self._state.items():
            out[cam_id] = {
                "status": st.status,
                "detail": st.detail,
                "candidates": list(st.candidates),
                "offline_for": (int(now - st.first_unhealthy_at)
                                if st.first_unhealthy_at else 0),
                "next_search_in": (max(0, int(st.next_scan_at - now))
                                   if st.next_scan_at else 0),
                "searches": st.scans,
            }
        return {"enabled": bool(_cfg("AUTO_DISCOVERY_ENABLED", True)),
                "auto_apply": bool(_cfg("AUTO_DISCOVERY_AUTO_APPLY", True)),
                "searching": self._scanning,
                "cameras": out}
