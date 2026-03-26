// ===== SSE Detection Stream =====
let evtSource = null;
let currentMode = "yolo";

function connectSSE() {
    if (evtSource) evtSource.close();
    evtSource = new EventSource("/stream/events");

    evtSource.addEventListener("detections", (e) => {
        const data = JSON.parse(e.data);
        currentMode = data.mode;
        updateDetections(data.detections, data.mode);
        updateFPS(data.fps);
        updateMode(data.mode);
    });

    evtSource.onerror = () => {
        updateStatus(false);
        setTimeout(connectSSE, 3000);
    };

    evtSource.onopen = () => {
        updateStatus(true);
    };
}

// ===== Status =====

function updateStatus(connected) {
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("status-text");
    dot.className = "status-dot " + (connected ? "connected" : "disconnected");
    text.textContent = connected ? "Connected" : "Disconnected";
}

function updateFPS(fps) {
    document.getElementById("fps-badge").textContent = fps.toFixed(1) + " FPS";
    document.getElementById("stat-fps").textContent = fps.toFixed(1);
}

function updateMode(mode) {
    currentMode = mode;
    const indicator = document.getElementById("mode-indicator");
    indicator.className = "mode-indicator mode-" + mode;
    const modeLabels = { yolo: "YOLO DETECTION", facenet: "FACE RECOGNITION", both: "YOLO + FACE" };
    const modeStats = { yolo: "YOLO", facenet: "FaceNet", both: "Both" };
    const modeTitles = { yolo: "Object Detections", facenet: "Face Detections", both: "All Detections" };
    indicator.textContent = modeLabels[mode] || mode.toUpperCase();
    document.getElementById("stat-mode").textContent = modeStats[mode] || mode;
    document.getElementById("btn-yolo").classList.toggle("active", mode === "yolo");
    document.getElementById("btn-facenet").classList.toggle("active", mode === "facenet");
    document.getElementById("btn-both").classList.toggle("active", mode === "both");
    document.getElementById("btn-yolo").classList.remove("switching");
    document.getElementById("btn-facenet").classList.remove("switching");
    document.getElementById("btn-both").classList.remove("switching");
    document.getElementById("det-card-title").textContent = modeTitles[mode] || "Detections";
}

// ===== Detections =====

function updateDetections(detections, mode) {
    const list = document.getElementById("detection-list");
    const countBadge = document.getElementById("det-count");

    if (!detections || detections.length === 0) {
        list.innerHTML = '<li class="no-data">No detections</li>';
        countBadge.textContent = "0";
        return;
    }

    countBadge.textContent = detections.length;

    if (mode === "both") {
        // Split into yolo and face detections
        const yoloDets = detections.filter(d => d.type === "yolo" || d.label);
        const faceDets = detections.filter(d => d.type === "face" || d.name);
        let html = "";

        if (yoloDets.length > 0) {
            const grouped = {};
            yoloDets.forEach(d => {
                const label = d.label || "?";
                if (!grouped[label]) grouped[label] = { count: 0, maxConf: 0 };
                grouped[label].count++;
                grouped[label].maxConf = Math.max(grouped[label].maxConf, d.confidence || 0);
            });
            html += Object.entries(grouped).map(([label, info]) => {
                const conf = (info.maxConf * 100).toFixed(0) + "%";
                const countStr = info.count > 1 ? ` x${info.count}` : "";
                return `<li><span class="det-label">${label}${countStr}</span><span class="det-conf">${conf}</span></li>`;
            }).join("");
        }

        if (faceDets.length > 0) {
            html += faceDets.map(d => {
                const name = d.name || "Unknown";
                const nameClass = name !== "Unknown" ? "det-name" : "det-unknown";
                const conf = d.confidence !== undefined ? (d.confidence * 100).toFixed(0) + "%" : "-";
                return `<li><span class="${nameClass}">${name}</span><span class="det-conf">${conf}</span></li>`;
            }).join("");
        }

        list.innerHTML = html || '<li class="no-data">No detections</li>';
    } else if (mode === "facenet") {
        list.innerHTML = detections.map(d => {
            const name = d.name || "Unknown";
            const isKnown = name !== "Unknown";
            const conf = d.confidence !== undefined ? (d.confidence * 100).toFixed(0) + "%" : "-";
            const nameClass = isKnown ? "det-name" : "det-unknown";
            return `<li><span class="${nameClass}">${name}</span><span class="det-conf">${conf}</span></li>`;
        }).join("");
    } else {
        const grouped = {};
        detections.forEach(d => {
            const label = d.label || "?";
            if (!grouped[label]) grouped[label] = { count: 0, maxConf: 0 };
            grouped[label].count++;
            grouped[label].maxConf = Math.max(grouped[label].maxConf, d.confidence || 0);
        });
        list.innerHTML = Object.entries(grouped).map(([label, info]) => {
            const conf = (info.maxConf * 100).toFixed(0) + "%";
            const countStr = info.count > 1 ? ` x${info.count}` : "";
            return `<li><span class="det-label">${label}${countStr}</span><span class="det-conf">${conf}</span></li>`;
        }).join("");
    }
}

// ===== Recent Events (sidebar, compact) =====

async function loadEvents() {
    try {
        const resp = await fetch("/api/events?limit=15");
        const data = await resp.json();
        const tbody = document.getElementById("event-tbody");

        if (!data.events || data.events.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="no-data">No events yet</td></tr>';
            return;
        }

        tbody.innerHTML = data.events.map(ev => {
            const time = ev.timestamp ? ev.timestamp.substring(11, 19) || ev.timestamp : "-";
            const type = ev.event_type || "-";
            const typeClass = type === "detection" ? "detection" : "face";
            let detail = "";
            if (type === "detection") {
                const conf = ev.confidence ? (ev.confidence * 100).toFixed(0) + "%" : "";
                detail = `${ev.object_type || "-"} ${conf}`;
            } else if (type === "face") {
                detail = `${ev.person_name || "-"} (${ev.distance || "-"})`;
            }
            const hasImage = ev.image_path ? "clickable" : "";
            const imgDir = type === "face" ? "events" : "roi";
            return `<tr class="${hasImage}" onclick="${ev.image_path ? `openLightboxFromEvent('${imgDir}', '${ev.image_path}', '${type}', '${detail}', '${ev.timestamp || ""}')` : ""}">
                <td>${time}</td>
                <td><span class="event-type ${typeClass}">${type}</span></td>
                <td>${detail}</td>
            </tr>`;
        }).join("");
    } catch (e) {
        console.error("Failed to load events:", e);
    }
}

// ===== Events Page (full browser) =====

let eventsFilter = "all";
let eventsOffset = 0;
const eventsPageSize = 24;
let lastKnownEventCount = 0;
let eventsAutoRefresh = true;
let eventsDateFrom = "";
let eventsDateTo = "";

async function loadEventsPage() {
    const grid = document.getElementById("events-grid");
    grid.innerHTML = '<div class="no-data">Loading...</div>';

    try {
        let url = `/api/events?limit=${eventsPageSize}&offset=${eventsOffset}&event_type=${eventsFilter}`;
        if (eventsDateFrom) url += `&date_from=${eventsDateFrom}`;
        if (eventsDateTo) url += `&date_to=${eventsDateTo}`;
        const resp = await fetch(url);
        const data = await resp.json();

        // Update paging info
        const total = data.total || 0;
        const showing = Math.min(eventsOffset + eventsPageSize, total);
        document.getElementById("events-showing").textContent = `${eventsOffset + 1}-${showing} of ${total}`;
        document.getElementById("events-prev").disabled = eventsOffset === 0;
        document.getElementById("events-next").disabled = eventsOffset + eventsPageSize >= total;

        if (!data.events || data.events.length === 0) {
            grid.innerHTML = '<div class="no-data">No events found</div>';
            return;
        }

        grid.innerHTML = data.events.map(ev => {
            const type = ev.event_type || "unknown";
            const typeClass = type === "detection" ? "detection" : "face";
            const time = ev.timestamp || "-";
            const imgDir = type === "face" ? "events" : "roi";
            const imgUrl = ev.image_path ? `/images/${imgDir}/${ev.image_path}` : null;
            const compositeId = `${type}_${ev.id}`;

            let title = "";
            let subtitle = "";
            if (type === "detection") {
                title = ev.object_type || "-";
                subtitle = ev.confidence ? (ev.confidence * 100).toFixed(0) + "% confidence" : "";
            } else {
                title = ev.person_name || "Unknown";
                subtitle = ev.distance ? `distance: ${ev.distance}` : "";
                if (ev.is_known) subtitle += " (known)";
            }

            return `<div class="event-card ${imgUrl ? 'has-image' : ''}" data-id="${compositeId}">
                <div class="event-card-select">
                    <input type="checkbox" class="event-checkbox" data-id="${compositeId}" onclick="event.stopPropagation(); updateSelectionBar()">
                </div>
                <div class="event-card-img" onclick="if('${imgUrl}' !== 'null') openLightbox('${imgUrl}', '${title}', '${subtitle}', '${time}')">
                    ${imgUrl
                        ? `<img src="${imgUrl}" alt="${title}" loading="lazy">`
                        : `<div class="event-card-noimg">No Image</div>`}
                    <span class="event-type-badge ${typeClass}">${type}</span>
                </div>
                <div class="event-card-body">
                    <div class="event-card-title">${title}</div>
                    <div class="event-card-sub">${subtitle}</div>
                    <div class="event-card-time">${time}</div>
                </div>
            </div>`;
        }).join("");
    } catch (e) {
        grid.innerHTML = '<div class="no-data">Failed to load events</div>';
        console.error("Failed to load events page:", e);
    }
}

// ===== Selection & Delete =====

function updateSelectionBar() {
    const checked = document.querySelectorAll(".event-checkbox:checked");
    const bar = document.getElementById("selection-bar");
    const count = document.getElementById("selected-count");

    if (checked.length > 0) {
        bar.classList.add("visible");
        count.textContent = checked.length;
    } else {
        bar.classList.remove("visible");
    }
}

function selectAllEvents() {
    const checkboxes = document.querySelectorAll(".event-checkbox");
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    checkboxes.forEach(cb => cb.checked = !allChecked);
    updateSelectionBar();
}

function clearSelection() {
    document.querySelectorAll(".event-checkbox").forEach(cb => cb.checked = false);
    updateSelectionBar();
}

async function deleteSelected() {
    const checked = document.querySelectorAll(".event-checkbox:checked");
    if (checked.length === 0) return;

    if (!confirm(`Delete ${checked.length} event(s)? This cannot be undone.`)) return;

    const ids = Array.from(checked).map(cb => cb.dataset.id);

    try {
        const resp = await fetch("/api/events/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ids })
        });
        if (resp.ok) {
            const data = await resp.json();
            console.log(`Deleted ${data.deleted} events`);
            clearSelection();
            loadEventsPage();
            loadEvents(); // refresh sidebar too
        }
    } catch (e) {
        console.error("Failed to delete events:", e);
    }
}

// ===== Lightbox =====

function openLightbox(imgUrl, title, subtitle, time) {
    const lb = document.getElementById("lightbox");
    document.getElementById("lightbox-img").src = imgUrl;
    document.getElementById("lightbox-info").innerHTML = `
        <strong>${title}</strong>
        <span>${subtitle}</span>
        <small>${time}</small>
    `;
    lb.classList.add("active");
}

function openLightboxFromEvent(imgDir, imagePath, type, detail, time) {
    if (!imagePath) return;
    const imgUrl = `/images/${imgDir}/${imagePath}`;
    openLightbox(imgUrl, detail, type, time);
}

function closeLightbox(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById("lightbox").classList.remove("active");
}

// ===== Date Filter =====

function applyDateFilter() {
    eventsDateFrom = document.getElementById("filter-date-from").value || "";
    eventsDateTo = document.getElementById("filter-date-to").value || "";
    eventsOffset = 0;
    loadEventsPage();
}

function clearDateFilter() {
    document.getElementById("filter-date-from").value = "";
    document.getElementById("filter-date-to").value = "";
    eventsDateFrom = "";
    eventsDateTo = "";
    eventsOffset = 0;
    loadEventsPage();
}

// ===== New Events Checker =====

async function checkNewEvents() {
    try {
        const resp = await fetch(`/api/events?limit=1&event_type=${eventsFilter}`);
        const data = await resp.json();
        const total = data.total || 0;

        // Update tab badge
        const badge = document.getElementById("events-badge");
        if (total > lastKnownEventCount && lastKnownEventCount > 0) {
            const newCount = total - lastKnownEventCount;
            badge.textContent = newCount;
            badge.classList.add("visible");
        }

        // Auto-refresh events grid if on the events tab and on first page
        const eventsTab = document.getElementById("tab-events");
        if (eventsTab.classList.contains("active") && eventsAutoRefresh && eventsOffset === 0) {
            if (total > lastKnownEventCount) {
                loadEventsPage();
            }
        }

        lastKnownEventCount = total;
    } catch (e) {
        // silent
    }
}

// ===== Tabs =====

function switchTab(tabName) {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === tabName);
    });
    document.querySelectorAll(".tab-content").forEach(content => {
        content.classList.toggle("active", content.id === "tab-" + tabName);
    });

    if (tabName === "events") {
        // Clear new events badge
        const badge = document.getElementById("events-badge");
        badge.classList.remove("visible");
        badge.textContent = "0";
        loadEventsPage();
    }
    if (tabName === "faces") {
        loadFaces();
    }
}

// ===== Face Enrollment =====

async function loadFaces() {
    const grid = document.getElementById("faces-grid");
    grid.innerHTML = '<div class="no-data">Loading...</div>';

    try {
        const resp = await fetch("/api/faces");
        const data = await resp.json();

        if (!data.people || data.people.length === 0) {
            grid.innerHTML = '<div class="no-data">No enrolled faces. Use the form above to add someone.</div>';
            return;
        }

        grid.innerHTML = data.people.map(p => {
            const thumb = p.thumbnail || "";
            const imgHtml = thumb
                ? `<img src="${thumb}" alt="${p.name}" loading="lazy">`
                : `<div class="face-card-noimg">No Photo</div>`;

            const imagesHtml = p.images.map(img =>
                `<div class="face-thumb">
                    <img src="/images/faces/${img}" alt="${img}" onclick="openLightbox('/images/faces/${img}', '${p.name}', '${img}', '')">
                    <button class="face-thumb-del" onclick="event.stopPropagation(); deletePersonImage('${p.name}', '${img}')" title="Delete image">&times;</button>
                </div>`
            ).join("");

            return `<div class="face-card" onclick="toggleFaceCard(this)">
                <div class="face-card-header">
                    <div class="face-card-avatar">${imgHtml}</div>
                    <div class="face-card-info">
                        <div class="face-card-name">${p.name}</div>
                        <div class="face-card-count">${p.image_count} photo${p.image_count !== 1 ? 's' : ''}</div>
                    </div>
                    <button class="btn-delete-person" onclick="event.stopPropagation(); deletePerson('${p.name}')" title="Delete person">&times;</button>
                </div>
                <div class="face-card-images">${imagesHtml}</div>
            </div>`;
        }).join("");
    } catch (e) {
        grid.innerHTML = '<div class="no-data">Failed to load faces</div>';
        console.error("Failed to load faces:", e);
    }
}

function toggleFaceCard(card) {
    card.classList.toggle("expanded");
}

async function enrollFace(event) {
    event.preventDefault();

    const name = document.getElementById("enroll-name").value.trim();
    const files = document.getElementById("enroll-files").files;
    const status = document.getElementById("enroll-status");
    const btn = document.getElementById("btn-enroll");

    if (!name || files.length === 0) {
        status.textContent = "Please provide a name and at least one photo.";
        status.className = "enroll-status error";
        return;
    }

    btn.disabled = true;
    btn.textContent = "Enrolling...";
    status.textContent = "";

    const cropFaces = document.getElementById("enroll-crop").checked;

    const formData = new FormData();
    formData.append("name", name);
    formData.append("crop_faces", cropFaces);
    for (const file of files) {
        formData.append("files", file);
    }

    try {
        const resp = await fetch("/api/faces/enroll", { method: "POST", body: formData });
        const data = await resp.json();

        if (resp.ok) {
            status.textContent = `Enrolled ${data.added.length} photo(s) for "${data.name}" (${data.total_images} total)`;
            status.className = "enroll-status success";
            document.getElementById("enroll-form").reset();
            loadFaces();
        } else {
            status.textContent = data.error || "Enrollment failed";
            status.className = "enroll-status error";
        }
    } catch (e) {
        status.textContent = "Network error during enrollment";
        status.className = "enroll-status error";
        console.error("Enroll error:", e);
    } finally {
        btn.disabled = false;
        btn.textContent = "Enroll";
    }
}

async function deletePerson(name) {
    if (!confirm(`Delete "${name}" and all their photos?`)) return;

    try {
        const resp = await fetch(`/api/faces/${encodeURIComponent(name)}`, { method: "DELETE" });
        if (resp.ok) {
            loadFaces();
        } else {
            const data = await resp.json();
            alert(data.error || "Failed to delete");
        }
    } catch (e) {
        console.error("Delete person error:", e);
    }
}

async function deletePersonImage(name, filename) {
    if (!confirm(`Delete this photo from "${name}"?`)) return;

    try {
        const resp = await fetch(`/api/faces/${encodeURIComponent(name)}/image/${encodeURIComponent(filename)}`, { method: "DELETE" });
        if (resp.ok) {
            loadFaces();
        } else {
            const data = await resp.json();
            alert(data.error || "Failed to delete image");
        }
    } catch (e) {
        console.error("Delete image error:", e);
    }
}

// ===== Mode Switching =====

async function setMode(mode) {
    const btnYolo = document.getElementById("btn-yolo");
    const btnFace = document.getElementById("btn-facenet");
    if (mode === "yolo") btnYolo.classList.add("switching");
    else btnFace.classList.add("switching");

    try {
        const resp = await fetch("/api/ai/mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode })
        });
        if (resp.ok) updateMode(mode);
    } catch (e) {
        console.error("Failed to set mode:", e);
        btnYolo.classList.remove("switching");
        btnFace.classList.remove("switching");
    }
}

// ===== Status Polling =====

async function pollStatus() {
    try {
        const resp = await fetch("/api/status");
        const data = await resp.json();
        updateStatus(data.camera_connected);
        if (data.ai_fps) updateFPS(data.ai_fps);
        if (data.ai_mode) updateMode(data.ai_mode);

        const info = document.getElementById("camera-info");
        if (info && data.frame_size) {
            info.textContent = `${data.frame_size[0]}x${data.frame_size[1]} | ${data.camera_type.toUpperCase()}`;
        }
        document.getElementById("stat-resolution").textContent =
            data.frame_size ? `${data.frame_size[0]}x${data.frame_size[1]}` : "-";
        document.getElementById("stat-camera").textContent =
            data.camera_type ? data.camera_type.toUpperCase() : "-";
    } catch (e) {
        updateStatus(false);
    }
}

// ===== Init =====

document.addEventListener("DOMContentLoaded", () => {
    connectSSE();
    loadEvents();
    pollStatus();

    setInterval(loadEvents, 10000);
    setInterval(pollStatus, 5000);
    setInterval(checkNewEvents, 5000);
    // Initialize event count baseline
    checkNewEvents();

    // Tab navigation
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    // Event filters
    document.querySelectorAll(".filter-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            eventsFilter = btn.dataset.filter;
            eventsOffset = 0;
            loadEventsPage();
        });
    });

    // Pagination
    document.getElementById("events-prev").addEventListener("click", () => {
        eventsOffset = Math.max(0, eventsOffset - eventsPageSize);
        loadEventsPage();
    });
    document.getElementById("events-next").addEventListener("click", () => {
        eventsOffset += eventsPageSize;
        loadEventsPage();
    });

    // Auto-refresh toggle
    document.getElementById("auto-refresh-check").addEventListener("change", (e) => {
        eventsAutoRefresh = e.target.checked;
    });

    // Close lightbox on Escape
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeLightbox();
    });
});
