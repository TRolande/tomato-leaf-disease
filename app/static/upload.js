/* Page: Upload photos. Uses /api/model (classes), /api/upload and /api/uploads. */
(() => {
  "use strict";
  const { $, el, api, prettyName, show, message, clearMessage, formatWhen, USERNAME_PATTERN, USERNAME_KEY } = window.TLC;
  let classNames = [];
  let chosenFiles = [];
  const thumbnails = new Map();       // file -> object URL of its preview
  const waitingCountNodes = {};       // class name -> span showing how many photos wait

  function thumbnailFor(file) {
    if (!thumbnails.has(file)) thumbnails.set(file, URL.createObjectURL(file));
    return thumbnails.get(file);
  }

  function forgetFile(file) {
    if (thumbnails.has(file)) {
      URL.revokeObjectURL(thumbnails.get(file));
      thumbnails.delete(file);
    }
  }

  function renderChoices() {
    for (const key of Object.keys(waitingCountNodes)) delete waitingCountNodes[key];
    $("classChoice").replaceChildren(...classNames.map((name, index) => {
      const id = `class-${index}`;
      const countNode = el("span", { class: "pill-count", text: "" });
      waitingCountNodes[name] = countNode;
      return el("span", null,
        el("input", { type: "radio", name: "photoClass", id, value: name, checked: index === 0 }),
        el("label", { for: id }, prettyName(name), countNode));
    }));
  }

  function selectedClass() {
    const checked = document.querySelector('input[name="photoClass"]:checked');
    return checked ? checked.value : classNames[0];
  }

  function renderFiles() {
    $("fileList").replaceChildren(...chosenFiles.map((file, index) =>
      el("li", null,
        el("img", { class: "thumb", src: thumbnailFor(file), alt: "" }),
        el("span", { class: "name", text: file.name }),
        el("span", { class: "muted small", text: TLC.formatBytes(file.size) }),
        el("button", { type: "button", text: "Remove", onclick: () => {
          forgetFile(file);
          chosenFiles.splice(index, 1);
          renderFiles();
        } }))));
    const any = chosenFiles.length > 0;
    $("uploadButton").disabled = !any;
    $("clearButton").disabled = !any;
    $("uploadButton").textContent = any ? `Upload ${chosenFiles.length} photo${chosenFiles.length === 1 ? "" : "s"}` : "Upload photos";
  }

  function addFiles(files) {
    const problems = [];
    for (const file of files) {
      if (!["image/jpeg", "image/png"].includes(file.type)) problems.push(`${file.name} is not a JPG or PNG photo.`);
      else if (file.size > TLC.UPLOAD_LIMIT_BYTES) problems.push(`${file.name} is larger than 10 MB.`);
      else chosenFiles.push(file);
    }
    if (problems.length) message($("uploadError"), problems.join(" ")); else clearMessage($("uploadError"));
    renderFiles();
  }

  function renderWaiting(counts) {
    const rows = classNames.map((name) => ({ name, pending: (counts[name] || { pending: 0 }).pending }));
    for (const row of rows) {
      if (waitingCountNodes[row.name]) waitingCountNodes[row.name].textContent = `${row.pending} waiting`;
    }
    const total = rows.reduce((sum, row) => sum + row.pending, 0);
    $("waitingNote").textContent = total
      ? `${total} photo${total === 1 ? "" : "s"} waiting. Retrain on the Retrain page.`
      : "No photos are waiting. Add some to enable retraining.";
    const largest = Math.max(1, ...rows.map((row) => row.pending));
    $("waitingList").replaceChildren(...rows.map((row) => el("li", null,
      el("div", { class: "waiting-head" }, el("span", { text: prettyName(row.name) }), el("b", { text: String(row.pending) })),
      el("div", { class: "waiting-bar" }, el("span", { style: `width:${(row.pending / largest) * 100}%` })))));
  }

  function renderRecent(recent) {
    if (!recent.length) {
      $("recentBody").replaceChildren(el("tr", null, el("td", { colspan: "5", class: "empty", text: "No uploads yet." })));
      return;
    }
    $("recentBody").replaceChildren(...recent.map((upload) => el("tr", null,
      el("td", { class: "name", text: upload.filename }),
      el("td", { text: prettyName(upload.label) }),
      el("td", { text: upload.username }),
      el("td", { text: formatWhen(upload.uploaded_at) }),
      el("td", null, el("span", { class: "chip " + (upload.status === "pending" ? "is-waiting" : "is-used"), text: upload.status === "pending" ? "Waiting" : "Used" })))));
  }

  async function refresh() {
    try {
      const uploads = await api("/api/uploads");
      renderWaiting(uploads.counts);
      renderRecent(uploads.recent);
    } catch (error) {
      message($("uploadError"), error.message);
    }
  }

  async function submit() {
    ["uploadOk", "uploadRejected", "uploadError"].forEach((id) => clearMessage($(id)));
    const username = $("usernameInput").value.trim();
    if (!USERNAME_PATTERN.test(username)) {
      message($("uploadError"), "Enter your username first: 2 to 32 letters, digits, dot, underscore or hyphen.");
      $("usernameInput").focus();
      return;
    }
    const label = selectedClass();
    const form = new FormData();
    form.append("label", label);
    form.append("username", username);
    chosenFiles.forEach((file) => form.append("files", file, file.name));

    const button = $("uploadButton");
    button.disabled = true;
    button.textContent = "Uploading...";
    try {
      const result = await api("/api/upload", { method: "POST", body: form });
      if (result.saved > 0) {
        message($("uploadOk"), `Saved ${result.saved} photo${result.saved === 1 ? "" : "s"} as ${prettyName(label)} by ${username}.`);
        chosenFiles.forEach(forgetFile);
        chosenFiles = [];
      }
      if (result.rejected.length) {
        $("uploadRejected").replaceChildren(
          el("div", { text: "These photos were not saved:" }),
          ...result.rejected.map((item) => el("div", { text: `${item.file}: ${item.reason}` })));
        show($("uploadRejected"), true);
      }
      renderFiles();
      await refresh();
    } catch (error) {
      message($("uploadError"), error.message);
    } finally {
      renderFiles();
    }
  }

  function setupDropzone(zone, input, onFiles) {
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => { onFiles([...input.files]); input.value = ""; });
    ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, (event) => {
      event.preventDefault(); zone.classList.add("is-over");
    }));
    ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, (event) => {
      event.preventDefault(); zone.classList.remove("is-over");
    }));
    zone.addEventListener("drop", (event) => onFiles([...event.dataTransfer.files]));
  }

  async function start() {
    try {
      classNames = (await api("/api/model")).classes;
      renderChoices();
    } catch (error) {
      message($("uploadError"), error.message);
    }
    try {
      const saved = localStorage.getItem(USERNAME_KEY);
      if (saved) $("usernameInput").value = saved;
    } catch (error) { /* storage blocked: the field starts empty */ }
    refresh();
  }

  $("usernameInput").addEventListener("change", () => {
    try { localStorage.setItem(USERNAME_KEY, $("usernameInput").value.trim()); } catch (error) { /* ignore */ }
  });
  setupDropzone($("dropzone"), $("photoInput"), addFiles);
  $("uploadButton").addEventListener("click", submit);
  $("clearButton").addEventListener("click", () => {
    chosenFiles.forEach(forgetFile);
    chosenFiles = [];
    renderFiles();
  });

  TLC.renderHeader("upload.html");
  TLC.renderFooter();
  start();
})();
