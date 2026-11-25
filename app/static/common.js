/* Shared by every page: the navigation bar, the status line, the API helper and small formatting helpers.
   Server text is always written with textContent, never as HTML. */
(() => {
  "use strict";

  const TLC = {};
  window.TLC = TLC;

  TLC.UPLOAD_LIMIT_BYTES = 10 * 1024 * 1024;
  TLC.USERNAME_PATTERN = /^[A-Za-z0-9_.-]{2,32}$/;
  TLC.USERNAME_KEY = "tomatoLeafCheck.username";
  TLC.PAGES = [
    { href: "index.html", label: "Check a photo" },
    { href: "dataset.html", label: "Dataset" },
    { href: "upload.html", label: "Upload photos" },
    { href: "retrain.html", label: "Retrain" }
  ];

  TLC.$ = (id) => document.getElementById(id);

  TLC.el = (tag, props, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (typeof value === "function" || typeof value === "boolean") node[key] = value;
      else node.setAttribute(key, value);
    }
    node.append(...children);
    return node;
  };

  TLC.show = (node, visible) => { node.hidden = !visible; };

  TLC.message = (node, text) => {
    node.textContent = text;
    TLC.show(node, true);
  };

  TLC.clearMessage = (node) => {
    node.textContent = "";
    TLC.show(node, false);
  };

  TLC.prettyName = (className) => {
    const words = className.replace(/^Tomato_/, "").replace(/_/g, " ").toLowerCase();
    return words.charAt(0).toUpperCase() + words.slice(1);
  };

  TLC.formatBytes = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  TLC.percent = (value) => (value == null ? "-" : (value * 100).toFixed(1) + "%");
  TLC.score = (value) => (value == null ? "-" : value.toFixed(3));

  TLC.formatUptime = (totalSeconds) => {
    const seconds = Math.max(0, Math.floor(totalSeconds));
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const pad = (n) => String(n).padStart(2, "0");
    return (days ? `${days} d ` : "") + `${pad(hours)}:${pad(minutes)}:${pad(seconds % 60)}`;
  };

  TLC.formatWhen = (unixSeconds) =>
    new Date(unixSeconds * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  TLC.api = async (path, options) => {
    let response;
    try {
      response = await fetch(path, options);
    } catch (networkError) {
      throw new Error("The server cannot be reached. Check the connection and try again.");
    }
    let body = null;
    try { body = await response.json(); } catch (parseError) { body = null; }
    if (!response.ok) {
      let detail = body && body.detail;
      if (Array.isArray(detail)) detail = detail.map((item) => (item && item.msg) || JSON.stringify(item)).join("; ");
      throw new Error(detail || `The server answered with status ${response.status}.`);
    }
    return body;
  };

  // ---------- navigation bar and status line ----------

  const LOGO_SVG =
    '<svg viewBox="0 0 26 26" aria-hidden="true" focusable="false">' +
    '<path d="M4 15C4 8.5 9 4 17.5 3.5 17.5 11 14 17 6.5 19.5 5 19.9 4 17.8 4 15Z" fill="#52b788"/>' +
    '<path d="M6.5 19.5 13 12.5" stroke="#14201a" stroke-width="1.6" stroke-linecap="round"/></svg>';

  TLC.health = null;
  let offlineBanner = null;
  let uptimeAtCheck = 0;
  let uptimeCheckedAt = 0;
  let isOnline = false;
  let statusDot, statusText, statusMeta, uptimeMeta;

  function renderStatusMeta() {
    if (!TLC.health) return;
    const parts = [
      `Model v${TLC.health.model_version}`,
      `accuracy ${TLC.percent(TLC.health.model_accuracy)}`,
      `up ${TLC.formatUptime(uptimeAtCheck + (Date.now() - uptimeCheckedAt) / 1000)}`
    ];
    statusMeta.textContent = parts.join(" · ");
  }

  async function refreshHealth() {
    try {
      TLC.health = await TLC.api("/api/health");
      uptimeAtCheck = TLC.health.uptime_seconds;
      uptimeCheckedAt = Date.now();
      isOnline = true;
      if (offlineBanner) offlineBanner.hidden = true;
      statusDot.className = "dot is-up";
      statusText.textContent = "Online";
      renderStatusMeta();
    } catch (error) {
      isOnline = false;
      if (offlineBanner) offlineBanner.hidden = false;
      statusDot.className = "dot is-down";
      statusText.textContent = "Offline";
      statusMeta.textContent = "";
    }
  }

  TLC.renderHeader = (activeHref) => {
    const header = document.getElementById("topbar");
    if (!header) return;
    statusDot = TLC.el("span", { class: "dot", "aria-hidden": "true" });
    statusText = TLC.el("span", { text: "Connecting" });
    statusMeta = TLC.el("span", { class: "status-meta" });

    const brand = TLC.el("a", { class: "brand", href: "index.html" });
    brand.innerHTML = LOGO_SVG;
    brand.append(TLC.el("span", { text: "Tomato Leaf Check" }));

    header.replaceChildren(TLC.el("div", { class: "container topbar-inner" },
      brand,
      TLC.el("nav", { class: "nav", "aria-label": "Pages" },
        ...TLC.PAGES.map((page) => TLC.el("a", {
          class: "nav-link" + (page.href === activeHref ? " is-active" : ""),
          href: page.href,
          "aria-current": page.href === activeHref ? "page" : null,
          text: page.label
        }))),
      TLC.el("div", { class: "status", "aria-live": "polite" }, statusDot, statusText, statusMeta)));

    offlineBanner = TLC.el("div", { class: "offline-banner", role: "alert", hidden: true },
      TLC.el("span", { text: "The server cannot be reached. Check that it is running, then try again." }),
      TLC.el("button", { type: "button", class: "btn btn-ghost btn-small", text: "Try again", onclick: () => refreshHealth() }));
    header.insertAdjacentElement("afterend", offlineBanner);

    let icon = document.querySelector('link[rel="icon"]');
    if (!icon) { icon = TLC.el("link", { rel: "icon" }); document.head.append(icon); }
    icon.href = "data:image/svg+xml," + encodeURIComponent(LOGO_SVG.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" '));

    refreshHealth();
    setInterval(refreshHealth, 5000);
    setInterval(() => { if (isOnline) renderStatusMeta(); }, 1000);
  };

  TLC.renderFooter = () => {
    const footer = document.getElementById("footer");
    if (footer) footer.replaceChildren(TLC.el("div", { class: "container", text: "Tomato Leaf Check. Four classes: Early blight, Late blight, Leaf mold and Healthy. Model trained on PlantVillage tomato leaf photos." }));
  };
})();
