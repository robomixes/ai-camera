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

    evtSource.addEventListener("alert", (e) => {
        const data = JSON.parse(e.data);
        handleAlert(data);
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

// ===== Bulk Delete =====

async function deleteAllFiltered() {
    const filterLabel = eventsFilter === "all" ? "ALL" : eventsFilter;
    const dateInfo = eventsDateFrom || eventsDateTo
        ? ` from ${eventsDateFrom || "start"} to ${eventsDateTo || "now"}`
        : "";

    if (!confirm(`Delete all "${filterLabel}" events${dateInfo}? This cannot be undone.`)) return;

    try {
        const resp = await fetch("/api/events/delete-all", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                event_type: eventsFilter,
                date_from: eventsDateFrom,
                date_to: eventsDateTo
            })
        });
        const data = await resp.json();
        alert(`Deleted ${data.deleted} event(s).`);
        eventsOffset = 0;
        loadEventsPage();
        loadEvents();
    } catch (e) {
        console.error("Bulk delete error:", e);
    }
}

async function deleteAllEvents() {
    if (!confirm("DELETE ALL EVENTS? This will remove every detection and face event from the database. This cannot be undone!")) return;
    if (!confirm("Are you really sure? This deletes EVERYTHING.")) return;

    try {
        const resp = await fetch("/api/events/delete-all", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ event_type: "all" })
        });
        const data = await resp.json();
        alert(`Deleted ${data.deleted} event(s).`);
        eventsOffset = 0;
        loadEventsPage();
        loadEvents();
    } catch (e) {
        console.error("Delete all error:", e);
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
    if (tabName === "settings") {
        loadSettings();
    }
}

// ===== Notifications =====

let notifHistory = [];
let unreadCount = 0;

function loadNotifPrefs() {
    const prefs = JSON.parse(localStorage.getItem("notifPrefs") || "{}");
    document.getElementById("notif-browser").checked = prefs.browser || false;
    document.getElementById("notif-sound").checked = prefs.sound !== false; // default true
}

function saveNotifPrefs() {
    const prefs = {
        browser: document.getElementById("notif-browser").checked,
        sound: document.getElementById("notif-sound").checked,
    };
    localStorage.setItem("notifPrefs", JSON.stringify(prefs));

    // Request browser notification permission if enabled
    if (prefs.browser && Notification.permission === "default") {
        Notification.requestPermission();
    }
}

function handleAlert(data) {
    const prefs = JSON.parse(localStorage.getItem("notifPrefs") || "{}");
    const now = new Date().toLocaleTimeString();

    // Add to history
    notifHistory.unshift({ ...data, time: now });
    if (notifHistory.length > 50) notifHistory.pop();
    unreadCount++;
    updateNotifBadge();
    renderNotifList();

    // Show toast
    showToast(data.title, data.detail);

    // Play sound
    if (prefs.sound !== false) {
        try {
            const audio = document.getElementById("alert-sound");
            if (audio) {
                audio.currentTime = 0;
                audio.play().catch(() => {});
            }
        } catch (e) {}
    }

    // Browser notification
    if (prefs.browser && Notification.permission === "granted") {
        try {
            new Notification(data.title, {
                body: data.detail,
                icon: "/api/snapshot",
                tag: data.type, // replaces previous notification of same type
            });
        } catch (e) {}
    }
}

function showToast(title, detail) {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<strong>${title}</strong><span>${detail}</span>`;
    container.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.classList.add("fade-out");
        setTimeout(() => toast.remove(), 300);
    }, 5000);

    // Click to dismiss
    toast.addEventListener("click", () => {
        toast.classList.add("fade-out");
        setTimeout(() => toast.remove(), 300);
    });
}

function updateNotifBadge() {
    const badge = document.getElementById("notif-badge");
    if (unreadCount > 0) {
        badge.textContent = unreadCount > 99 ? "99+" : unreadCount;
        badge.classList.add("visible");
    } else {
        badge.classList.remove("visible");
    }
}

function renderNotifList() {
    const list = document.getElementById("notif-list");
    if (notifHistory.length === 0) {
        list.innerHTML = '<li class="no-data">No alerts yet</li>';
        return;
    }
    list.innerHTML = notifHistory.map(n => {
        const typeClass = n.type === "unknown_face" ? "alert-unknown" : n.type === "known_face" ? "alert-known" : "alert-person";
        return `<li class="notif-item ${typeClass}">
            <span class="notif-title">${n.title}</span>
            <span class="notif-detail">${n.detail}</span>
            <span class="notif-time">${n.time}</span>
        </li>`;
    }).join("");
}

function toggleNotifPanel() {
    const panel = document.getElementById("notif-panel");
    panel.classList.toggle("visible");
    if (panel.classList.contains("visible")) {
        unreadCount = 0;
        updateNotifBadge();
    }
}

function clearNotifHistory() {
    notifHistory = [];
    unreadCount = 0;
    updateNotifBadge();
    renderNotifList();
}

// Close panel on outside click
document.addEventListener("click", (e) => {
    const wrap = document.querySelector(".notif-bell-wrap");
    const panel = document.getElementById("notif-panel");
    if (wrap && !wrap.contains(e.target) && panel.classList.contains("visible")) {
        panel.classList.remove("visible");
    }
});

// ===== Settings =====

let originalSettings = {};

async function loadSettings() {
    const container = document.getElementById("settings-container");
    container.innerHTML = '<div class="no-data">Loading settings...</div>';
    document.getElementById("settings-status").textContent = "";

    try {
        const resp = await fetch("/api/settings");
        const data = await resp.json();
        originalSettings = {};

        let html = "";

        // Group runtime settings by category
        const runtimeByCategory = {};
        for (const [key, info] of Object.entries(data.runtime)) {
            const cat = info.category || "General";
            if (!runtimeByCategory[cat]) runtimeByCategory[cat] = [];
            runtimeByCategory[cat].push({ key, ...info });
            originalSettings[key] = info.value;
        }

        // Group readonly settings by category
        const readonlyByCategory = {};
        for (const [key, info] of Object.entries(data.readonly)) {
            const cat = info.category || "General";
            if (!readonlyByCategory[cat]) readonlyByCategory[cat] = [];
            readonlyByCategory[cat].push({ key, ...info });
        }

        // Render runtime settings
        html += '<h3 class="settings-section-title">Runtime Settings <span class="settings-hint">Changes apply immediately</span></h3>';
        for (const [cat, settings] of Object.entries(runtimeByCategory)) {
            html += `<div class="card settings-card">
                <div class="card-header">${cat}</div>
                <div class="card-body">`;
            for (const s of settings) {
                html += renderSettingRow(s, false);
            }
            html += `</div></div>`;
        }

        // Render readonly settings
        html += '<h3 class="settings-section-title">Server Settings <span class="settings-hint">Requires restart to change</span></h3>';
        for (const [cat, settings] of Object.entries(readonlyByCategory)) {
            html += `<div class="card settings-card">
                <div class="card-header">${cat}</div>
                <div class="card-body">`;
            for (const s of settings) {
                html += renderSettingRow(s, true);
            }
            html += `</div></div>`;
        }

        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = '<div class="no-data">Failed to load settings</div>';
        console.error("Failed to load settings:", e);
    }
}

function renderSettingRow(s, readonly) {
    const val = s.type === "list" ? (Array.isArray(s.value) ? s.value.join(", ") : s.value) : s.value;
    const disabledAttr = readonly ? "disabled" : "";
    const readonlyClass = readonly ? "readonly" : "";

    let inputHtml = "";
    if (s.type === "bool") {
        const checked = val ? "checked" : "";
        inputHtml = `<input type="checkbox" class="setting-input setting-checkbox" data-key="${s.key}" ${checked} ${disabledAttr}>`;
        return `<div class="setting-row ${readonlyClass}">
            <div class="setting-label">
                <span class="setting-key">${s.key}</span>
                <span class="setting-desc">${s.description || ''}</span>
            </div>
            <div class="setting-value">${inputHtml}</div>
        </div>`;
    } else if (s.type === "float" || s.type === "int") {
        const step = s.type === "float" ? "0.1" : "1";
        const min = s.min !== undefined ? `min="${s.min}"` : "";
        const max = s.max !== undefined ? `max="${s.max}"` : "";
        inputHtml = `<input type="number" class="setting-input" data-key="${s.key}" value="${val}" step="${step}" ${min} ${max} ${disabledAttr}>`;
        if (s.min !== undefined && s.max !== undefined) {
            inputHtml += `<span class="setting-range">${s.min} – ${s.max}</span>`;
        }
    } else if (s.type === "list") {
        inputHtml = `<input type="text" class="setting-input" data-key="${s.key}" value="${val}" ${disabledAttr} placeholder="comma-separated">`;
    } else {
        inputHtml = `<input type="text" class="setting-input" data-key="${s.key}" value="${val || ''}" ${disabledAttr}>`;
    }

    return `<div class="setting-row ${readonlyClass}">
        <div class="setting-label">
            <span class="setting-key">${s.key}</span>
            <span class="setting-desc">${s.description || ''}</span>
        </div>
        <div class="setting-value">${inputHtml}</div>
    </div>`;
}

async function saveSettings() {
    const inputs = document.querySelectorAll(".setting-input:not([disabled])");
    const changes = {};

    inputs.forEach(input => {
        const key = input.dataset.key;
        let val;
        if (input.type === "checkbox") {
            val = input.checked;
            if (val !== originalSettings[key]) {
                changes[key] = val;
            }
        } else {
            val = input.value;
            const orig = originalSettings[key];
            const origStr = Array.isArray(orig) ? orig.join(", ") : String(orig);
            if (val !== origStr) {
                changes[key] = val;
            }
        }
    });

    const statusEl = document.getElementById("settings-status");

    if (Object.keys(changes).length === 0) {
        statusEl.textContent = "No changes to save.";
        statusEl.className = "settings-status";
        return;
    }

    try {
        const resp = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(changes)
        });
        const data = await resp.json();

        if (data.errors && Object.keys(data.errors).length > 0) {
            const errMsgs = Object.entries(data.errors).map(([k, v]) => `${k}: ${v}`).join(", ");
            statusEl.textContent = `Saved with errors: ${errMsgs}`;
            statusEl.className = "settings-status error";
        } else {
            const count = Object.keys(data.updated).length;
            statusEl.textContent = `${count} setting(s) saved successfully.`;
            statusEl.className = "settings-status success";
            // Update originals
            for (const [k, v] of Object.entries(data.updated)) {
                originalSettings[k] = v;
            }
        }
    } catch (e) {
        statusEl.textContent = "Failed to save settings.";
        statusEl.className = "settings-status error";
        console.error("Save settings error:", e);
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
    loadNotifPrefs();

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
