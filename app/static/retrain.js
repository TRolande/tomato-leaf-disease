/* Page: Retrain. Uses /api/uploads, /api/retrain, /api/retrain/status and /api/model. */
(() => {
  "use strict";
  const { $, el, api, percent, score, show, message, clearMessage, formatWhen } = window.TLC;
  let pollTimer = null;
  const validationByEpoch = new Map();     // epoch number -> validation accuracy, collected while polling
  const SVG_NS = "http://www.w3.org/2000/svg";

  function drawLiveChart() {
    const box = $("liveChartBox");
    const svg = $("liveChart");
    const epochs = [...validationByEpoch.keys()].sort((a, b) => a - b);
    if (!epochs.length) { show(box, false); return; }
    show(box, true);
    const width = 300, height = 90, pad = 8;
    const x = (index) => (epochs.length === 1 ? width / 2 : (index / (epochs.length - 1)) * (width - 2 * pad) + pad);
    const y = (value) => height - pad - Math.max(0, Math.min(1, value)) * (height - 2 * pad);
    const points = epochs.map((epoch, index) => [x(index), y(validationByEpoch.get(epoch))]);
    const nodes = [];
    for (const level of [0, 0.5, 1]) {
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("class", "grid-line");
      line.setAttribute("x1", "0"); line.setAttribute("x2", String(width));
      line.setAttribute("y1", String(y(level))); line.setAttribute("y2", String(y(level)));
      nodes.push(line);
    }
    const trace = document.createElementNS(SVG_NS, "polyline");
    trace.setAttribute("class", "trace");
    trace.setAttribute("points", points.map((p) => p.join(",")).join(" "));
    nodes.push(trace);
    points.forEach(([px, py]) => {
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("class", "dot-point");
      dot.setAttribute("cx", px); dot.setAttribute("cy", py); dot.setAttribute("r", "3");
      nodes.push(dot);
    });
    svg.replaceChildren(...nodes);
    const last = validationByEpoch.get(epochs[epochs.length - 1]);
    $("liveValue").textContent = `epoch ${epochs[epochs.length - 1]}: ${percent(last)}`;
  }

  function elapsedText(startedAtSeconds) {
    const seconds = Math.max(0, Math.floor(Date.now() / 1000 - startedAtSeconds));
    const pad = (n) => String(n).padStart(2, "0");
    return `Running for ${Math.floor(seconds / 60)}:${pad(seconds % 60)}`;
  }

  const MEASURES = [
    { key: "accuracy", label: "Accuracy", percentage: true, higherIsBetter: true },
    { key: "loss", label: "Loss", higherIsBetter: false },
    { key: "precision", label: "Precision", higherIsBetter: true },
    { key: "recall", label: "Recall", higherIsBetter: true },
    { key: "f1", label: "F1 score", higherIsBetter: true },
    { key: "roc_auc", label: "ROC AUC", higherIsBetter: true }
  ];

  function change(measure, before, after) {
    if (before == null || after == null) return { text: "-", good: null };
    const difference = after - before;
    if (Math.abs(difference) < 1e-9) return { text: "no change", good: null };
    const sign = difference > 0 ? "+" : "-";
    const text = measure.percentage
      ? `${sign}${(Math.abs(difference) * 100).toFixed(1)} points`
      : `${sign}${Math.abs(difference).toFixed(3)}`;
    return { text, good: measure.higherIsBetter ? difference > 0 : difference < 0 };
  }

  function showResult(result) {
    const count = result.new_images;
    $("resultSummary").textContent = `Trained on ${count} new photo${count === 1 ? "" : "s"}. The new version is live.`;
    $("resultBody").replaceChildren(...MEASURES.map((measure) => {
      const before = result.before[measure.key];
      const after = result.after[measure.key];
      const delta = change(measure, before, after);
      const cls = delta.good === true ? "up" : delta.good === false ? "down" : "";
      return el("tr", null,
        el("td", { class: "name", text: measure.label }),
        el("td", { class: "num", text: measure.percentage ? percent(before) : score(before) }),
        el("td", { class: "num", text: measure.percentage ? percent(after) : score(after) }),
        el("td", { class: "num " + cls, text: delta.text }));
    }));
    show($("resultSection"), true);
  }

  function showProgress(status) {
    show($("progress"), true);
    const epoch = status.epoch || 0;
    const total = status.total_epochs || 1;
    const done = Math.min(100, Math.round((epoch / total) * 100));
    $("progressEpoch").textContent = epoch ? `Epoch ${epoch} of ${total}` : "Starting";
    $("progressPercent").textContent = `${done}%`;
    $("progressFill").style.width = `${done}%`;
    $("progressMessage").textContent = status.message || "";
    $("progressElapsed").textContent = status.started_at ? elapsedText(status.started_at) : "";
    // the chart is rebuilt from the epochs saved in the database, so it survives a reload
    if (Array.isArray(status.epochs)) {
      validationByEpoch.clear();
      for (const row of status.epochs) {
        if (row.epoch && row.val_accuracy != null) validationByEpoch.set(row.epoch, row.val_accuracy);
      }
      drawLiveChart();
    }
  }

  function stopPolling() {
    if (pollTimer !== null) clearInterval(pollTimer);
    pollTimer = null;
  }

  async function poll() {
    let status;
    try {
      status = await api("/api/retrain/status");
    } catch (error) {
      return;
    }
    if (status.status === "running") {
      $("retrainButton").disabled = true;
      showProgress(status);
      return;
    }
    stopPolling();
    show($("progress"), false);
    if (status.status === "done") {
      showResult(status.result);
      loadVersions();
    } else if (status.status === "error" || status.status === "interrupted") {
      message($("retrainError"), status.message || "Retraining did not finish.");
    }
    loadWaiting();
  }

  function startPolling() {
    if (pollTimer === null) pollTimer = setInterval(poll, 2000);
    poll();
  }

  async function loadWaiting() {
    try {
      const uploads = await api("/api/uploads");
      $("waitingCount").textContent = uploads.pending;
      $("waitingCaption").textContent = uploads.pending
        ? "photos will be used in the next run"
        : "No photos are waiting. Upload some on the Upload page first.";
      $("retrainButton").disabled = uploads.pending === 0 || pollTimer !== null;
    } catch (error) {
      message($("retrainError"), error.message);
    }
  }

  async function loadVersions() {
    try {
      const model = await api("/api/model");
      $("versionsList").replaceChildren(...model.history.slice().reverse().map((version) => el("li", { class: version.is_current ? "is-live" : "" },
        el("div", { class: "version-head" },
          el("span", { text: `Version ${version.id}` }),
          el("span", { class: "chip " + (version.is_current ? "is-live" : "is-old"), text: version.is_current ? "Live" : "Earlier" })),
        el("div", { class: "version-meta", text:
          `accuracy ${percent(version.accuracy)} · F1 score ${score(version.f1)} · ` +
          `${version.n_new_images} new photo${version.n_new_images === 1 ? "" : "s"} · ${formatWhen(version.created_at)}` }))));
    } catch (error) {
      message($("retrainError"), error.message);
    }
  }

  async function startRetrain() {
    clearMessage($("retrainError"));
    show($("confirmBox"), false);
    $("retrainButton").disabled = true;
    validationByEpoch.clear();
    drawLiveChart();
    try {
      await api("/api/retrain", { method: "POST" });
      show($("resultSection"), false);
      showProgress({ epoch: 0, total_epochs: 1, message: "Starting" });
      startPolling();
    } catch (error) {
      message($("retrainError"), error.message);
      loadWaiting();
    }
  }

  $("retrainButton").addEventListener("click", () => {
    const count = Number($("waitingCount").textContent) || 0;
    $("confirmText").textContent =
      `Retrain on ${count} waiting photo${count === 1 ? "" : "s"}? When the run finishes, the new version replaces the live model.`;
    show($("confirmBox"), true);
    $("confirmStart").focus();
  });
  $("confirmStart").addEventListener("click", startRetrain);
  $("confirmCancel").addEventListener("click", () => { show($("confirmBox"), false); $("retrainButton").focus(); });

  TLC.renderHeader("retrain.html");
  TLC.renderFooter();
  loadWaiting();
  loadVersions();
  startPolling();   // shows a run that is already going, and the last result, then stops polling
})();
