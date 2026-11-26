/* Page: Dataset. Uses /api/insights. Each chart has a caption computed from the numbers. */
(() => {
  "use strict";
  const { $, api, prettyName } = window.TLC;
  const charts = [];

  function bar(canvasId, labels, datasets, legend) {
    return new Chart($(canvasId), {
      type: "bar",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: legend, position: "bottom", labels: { boxWidth: 12, usePointStyle: true } } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, grid: { color: "#e8e3d7" } }
        }
      }
    });
  }

  function extremes(values, labels) {
    const highest = values.indexOf(Math.max(...values));
    const lowest = values.indexOf(Math.min(...values));
    return { high: prettyName(labels[highest]), highValue: Math.max(...values), low: prettyName(labels[lowest]), lowValue: Math.min(...values) };
  }

  function render(data) {
    charts.forEach((chart) => chart.destroy());
    charts.length = 0;
    const names = data.classes.map(prettyName);
    const total = data.count.reduce((sum, n) => sum + n, 0);
    $("statPhotos").textContent = total.toLocaleString();
    $("statClasses").textContent = data.classes.length;
    $("statSample").textContent = data.sample_size;

    const count = Math.min(...data.count), most = Math.max(...data.count);
    charts.push(bar("chartCount", names, [{ label: "Training photos", data: data.count, backgroundColor: "#2d6a4f", borderRadius: 6 }], false));
    $("captionCount").textContent = count === most
      ? `Each class has ${count} training photos, so accuracy is a fair measure.`
      : `Class sizes range from ${count} to ${most} photos. Watch the recall of the smallest class.`;

    const channels = [
      { key: "mean_r", label: "Red", colour: "#b8554c" },
      { key: "mean_g", label: "Green", colour: "#4f8a5b" },
      { key: "mean_b", label: "Blue", colour: "#4a72a8" }
    ];
    charts.push(bar("chartColour", names, channels.map((c) => ({ label: c.label, data: data[c.key], backgroundColor: c.colour, borderRadius: 4 })), true));
    let widest = { label: "", range: -1 };
    for (const c of channels) {
      const range = Math.max(...data[c.key]) - Math.min(...data[c.key]);
      if (range > widest.range) widest = { label: c.label.toLowerCase(), range };
    }
    $("captionColour").textContent = `The classes differ by at most ${widest.range.toFixed(1)} points on any colour channel (largest in ${widest.label}). Colour alone does not separate them.`;

    charts.push(bar("chartBrightness", names, [{ label: "Median brightness", data: data.brightness, backgroundColor: "#8a8f6a", borderRadius: 6 }], false));
    const bright = extremes(data.brightness, data.classes);
    $("captionBrightness").textContent = `${bright.low} photos are the darkest (median ${bright.lowValue}) and ${bright.high} the brightest (median ${bright.highValue}). Lighting varies, so brightness changes are part of the augmentation.`;

    charts.push(bar("chartTexture", names, [{ label: "Median texture", data: data.texture, backgroundColor: "#7c5c9e", borderRadius: 6 }], false));
    const texture = extremes(data.texture, data.classes);
    $("captionTexture").textContent = `${texture.high} leaves have the highest texture (median ${texture.highValue}) and ${texture.low} the lowest (median ${texture.lowValue}).`;
  }

  async function load() {
    if (typeof Chart === "undefined") {
      TLC.message($("datasetError"), "The chart library could not be loaded. Check the connection and reload the page.");
      return;
    }
    try {
      render(await api("/api/insights"));
    } catch (error) {
      TLC.message($("datasetError"), error.message);
    }
  }

  TLC.renderHeader("dataset.html");
  TLC.renderFooter();
  load();
})();
