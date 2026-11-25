/* Page: Check a photo. Uses /api/predict. */
(() => {
  "use strict";
  const { $, el, api, percent, prettyName, show, message, clearMessage } = window.TLC;
  const UNCERTAIN_BELOW = 0.70;
  const GUIDE = {
    Tomato_healthy: "Even green tissue with no spots, rings or fuzzy growth. Keep watering at the soil and check the lower leaves once a week.",
    Tomato_Early_blight: "Dark brown spots with concentric rings, usually on older lower leaves. Remove the affected leaves and keep water off the foliage.",
    Tomato_Late_blight: "Water-soaked dark patches that spread quickly and turn brown or papery, mostly in cool, wet weather. Bag and remove infected leaves.",
    Tomato_Leaf_Mold: "Pale yellow patches on the upper surface, with olive-grey mould underneath, in humid air. Improve airflow and lower the humidity."
  };

  let chosenPhoto = null;
  let previewUrl = null;

  function problemWith(file) {
    if (!["image/jpeg", "image/png"].includes(file.type)) return `${file.name} is not a JPG or PNG photo.`;
    if (file.size > TLC.UPLOAD_LIMIT_BYTES) return `${file.name} is larger than 10 MB.`;
    return null;
  }

  function choosePhoto(file) {
    clearMessage($("checkError"));
    const problem = problemWith(file);
    if (problem) { message($("checkError"), problem); return; }
    chosenPhoto = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    $("preview").src = previewUrl;
    show($("preview"), true);
    show($("dropzoneEmpty"), false);
    $("fileName").textContent = file.name;
    $("checkButton").disabled = false;
    show($("resultBody"), false);
    show($("resultEmpty"), true);
  }

  function resetCheck() {
    chosenPhoto = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $("preview").removeAttribute("src");
    show($("preview"), false);
    show($("dropzoneEmpty"), true);
    $("fileName").textContent = "";
    $("checkButton").disabled = true;
    show($("resultBody"), false);
    show($("resultEmpty"), true);
    clearMessage($("checkError"));
    $("photoInput").focus();
  }

  function showResult(result) {
    const healthy = result.label === "Tomato_healthy";
    $("resultPhoto").textContent = chosenPhoto ? `${chosenPhoto.name} · ${TLC.formatBytes(chosenPhoto.size)}` : "";
    $("resultModel").textContent = TLC.health ? `Checked by model v${TLC.health.model_version}` : "";
    $("verdictTag").className = "verdict-tag" + (healthy ? "" : " is-disease");
    $("verdictTag").textContent = healthy ? "Healthy" : "Disease";
    $("verdictName").textContent = prettyName(result.label);
    $("verdictConfidence").textContent = percent(result.confidence);
    $("uncertainValue").textContent = percent(result.confidence);
    show($("uncertainNote"), result.confidence < UNCERTAIN_BELOW);
    $("verdictGuide").textContent = GUIDE[result.label] || "";

    const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
    $("probabilityRows").replaceChildren(...ranked.map(([name, probability], index) => {
      const top = index === 0;
      return el("div", { class: "prob-row" + (top ? " is-top" : "") + (top && !healthy ? " is-disease" : "") },
        el("span", { class: "prob-name", text: prettyName(name) }),
        el("div", { class: "prob-track" },
          el("div", { class: "prob-fill", style: `width:${(probability * 100).toFixed(1)}%` })),
        el("span", { class: "prob-value", text: percent(probability) }));
    }));
    show($("resultEmpty"), false);
    show($("resultBody"), true);
  }

  async function runCheck() {
    if (!chosenPhoto) return;
    const button = $("checkButton");
    button.disabled = true;
    button.textContent = "Checking...";
    clearMessage($("checkError"));
    try {
      const form = new FormData();
      form.append("file", chosenPhoto, chosenPhoto.name);
      showResult(await api("/api/predict", { method: "POST", body: form }));
    } catch (error) {
      message($("checkError"), error.message);
    } finally {
      button.disabled = false;
      button.textContent = "Check photo";
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

  setupDropzone($("dropzone"), $("photoInput"), (files) => { if (files[0]) choosePhoto(files[0]); });
  $("checkButton").addEventListener("click", runCheck);
  $("resetButton").addEventListener("click", resetCheck);
  document.addEventListener("paste", (event) => {
    const tag = event.target && event.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    const image = [...(event.clipboardData ? event.clipboardData.items : [])].find((item) => item.type.startsWith("image/"));
    if (image) choosePhoto(image.getAsFile());
  });

  TLC.renderHeader("index.html");
  TLC.renderFooter();
})();
