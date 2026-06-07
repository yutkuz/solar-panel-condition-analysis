const state = { file: null, objectUrl: null, busy: false };

const classLabels = {
  bird_drop: "Kuş pisliği",
  clean: "Temiz",
  crack_or_damage: "Çatlak veya hasar",
  dust: "Toz",
  snow: "Kar",
};

const elements = {
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  filePill: document.querySelector("#filePill"),
  fileName: document.querySelector("#fileName"),
  removeFile: document.querySelector("#removeFile"),
  analyzeButton: document.querySelector("#analyzeButton"),
  clearButton: document.querySelector("#clearButton"),
  detectorModel: document.querySelector("#detectorModel"),
  classifierModel: document.querySelector("#classifierModel"),
  threshold: document.querySelector("#threshold"),
  thresholdValue: document.querySelector("#thresholdValue"),
  thresholdControl: document.querySelector("#thresholdControl"),
  statusBox: document.querySelector("#statusBox"),
  statusTitle: document.querySelector("#statusTitle"),
  statusMessage: document.querySelector("#statusMessage"),
  results: document.querySelector("#results"),
  resultMessage: document.querySelector("#resultMessage"),
  originalPreview: document.querySelector("#originalPreview"),
  annotatedPreview: document.querySelector("#annotatedPreview"),
  timingGrid: document.querySelector("#timingGrid"),
  panelList: document.querySelector("#panelList"),
  panelCount: document.querySelector("#panelCount"),
  jsonDownload: document.querySelector("#jsonDownload"),
  imageDownload: document.querySelector("#imageDownload"),
  systemChip: document.querySelector("#systemChip"),
  systemText: document.querySelector("#systemText"),
};

function selectedMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function selectedLabel(select) {
  return select.selectedOptions[0].textContent.split("·")[0].trim();
}

function setFile(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showError("Desteklenmeyen dosya", "JPG, PNG veya WebP formatında bir fotoğraf seçin.");
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    showError("Dosya çok büyük", "Fotoğraf boyutu en fazla 15 MB olabilir.");
    return;
  }
  state.file = file;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  elements.fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  elements.filePill.classList.remove("hidden");
  elements.analyzeButton.disabled = false;
  elements.results.classList.add("hidden");
  hideStatus();
}

function clearAll(event) {
  if (event) event.stopPropagation();
  state.file = null;
  elements.fileInput.value = "";
  elements.filePill.classList.add("hidden");
  elements.analyzeButton.disabled = true;
  elements.results.classList.add("hidden");
  hideStatus();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
}

function showLoading() {
  const mode = selectedMode();
  const detector = selectedLabel(elements.detectorModel);
  const classifier = selectedLabel(elements.classifierModel);
  let detail = `${detector} ve ${classifier} hazırlanıyor.`;
  if (mode === "classification") detail = `${classifier} hazırlanıyor ve fotoğraf sınıflandırılıyor.`;
  if (mode === "detection") detail = `${detector} hazırlanıyor ve paneller tespit ediliyor.`;
  elements.statusBox.classList.remove("hidden", "error");
  elements.statusTitle.textContent = "Fotoğraf analiz ediliyor";
  elements.statusMessage.textContent = detail;
  elements.results.classList.add("hidden");
}

function showError(title, message) {
  elements.statusBox.classList.remove("hidden");
  elements.statusBox.classList.add("error");
  elements.statusTitle.textContent = title;
  elements.statusMessage.textContent = message;
}

function hideStatus() {
  elements.statusBox.classList.add("hidden");
  elements.statusBox.classList.remove("error");
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function probabilityRows(probabilities) {
  return Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => `
      <div class="probability">
        <span>${classLabels[name] || name}</span>
        <span class="bar"><i style="width:${(value * 100).toFixed(1)}%"></i></span>
        <span>%${(value * 100).toFixed(1)}</span>
      </div>
    `).join("");
}

function renderPanel(panel) {
  const detectionOnly = panel.classification_confidence === null;
  const confidence = detectionOnly ? panel.detection_confidence : panel.classification_confidence;
  const lowClass = panel.low_confidence ? " low" : "";
  const prefix = panel.low_confidence ? "Düşük güven" : "Güven";
  return `
    <article class="panel-card${detectionOnly ? " detection-only" : ""}">
      <img src="${panel.crop_url}" alt="${panel.panel_id}. panel kırpımı">
      <div class="panel-summary">
        <small>Panel ${String(panel.panel_id).padStart(2, "0")}</small>
        <h4>${detectionOnly ? "Güneş paneli" : panel.class_label}</h4>
        <span class="confidence${lowClass}">${prefix} · %${(confidence * 100).toFixed(1)}</span>
      </div>
      ${detectionOnly ? "" : `<div class="probabilities">${probabilityRows(panel.probabilities)}</div>`}
    </article>
  `;
}

function renderResults(data) {
  hideStatus();
  elements.resultMessage.textContent = data.message;
  elements.originalPreview.src = state.objectUrl;
  elements.annotatedPreview.src = `${data.annotated_image_url}?v=${Date.now()}`;
  elements.imageDownload.href = `/api/download/${data.request_id}/annotated.jpg`;
  elements.jsonDownload.href = `/api/download/${data.request_id}/result.json`;
  elements.panelCount.textContent = `${data.panel_count} panel`;

  const metrics = [];
  if (data.detector_label) metrics.push(metric("Tespit modeli", data.detector_label));
  if (data.classifier_label) metrics.push(metric("Sınıflandırma", data.classifier_label));
  metrics.push(metric("Cihaz", data.device.toUpperCase()));
  if (data.mode !== "classification") {
    metrics.push(metric("Tespit", `${data.timings.detection_seconds.toFixed(2)} sn`));
  }
  if (data.mode !== "detection") {
    metrics.push(metric("Sınıflandırma", `${data.timings.classification_seconds.toFixed(2)} sn`));
  }
  metrics.push(metric("Toplam", `${data.timings.total_seconds.toFixed(2)} sn`));
  elements.timingGrid.innerHTML = metrics.join("");

  elements.panelList.innerHTML = data.panels.length
    ? data.panels.map(renderPanel).join("")
    : '<div class="empty-panels">Bu güven eşiğinde panel bulunamadı. Eşiği düşürerek yeniden deneyebilirsiniz.</div>';
  elements.results.classList.remove("hidden");
  elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyze() {
  if (!state.file || state.busy) return;
  state.busy = true;
  elements.analyzeButton.disabled = true;
  showLoading();

  const form = new FormData();
  form.append("file", state.file);
  form.append("detector_model", elements.detectorModel.value);
  form.append("classifier_model", elements.classifierModel.value);
  form.append("mode", selectedMode());
  form.append("threshold", elements.threshold.value);

  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analiz tamamlanamadı.");
    renderResults(data);
  } catch (error) {
    showError("Analiz tamamlanamadı", error.message);
  } finally {
    state.busy = false;
    elements.analyzeButton.disabled = !state.file;
  }
}

function applyMode() {
  const mode = selectedMode();
  const classificationOnly = mode === "classification";
  const detectionOnly = mode === "detection";
  elements.detectorModel.disabled = classificationOnly;
  elements.classifierModel.disabled = detectionOnly;
  elements.thresholdControl.classList.toggle("disabled", classificationOnly);
  elements.threshold.disabled = classificationOnly;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const detectorModels = Object.values(data.models.detectors);
    const classifierModels = Object.values(data.models.classifiers);
    const allModels = [...detectorModels, ...classifierModels];
    const readyCount = allModels.filter(
      (model) => model.checkpoint_exists && model.dependency_installed
    ).length;
    elements.systemChip.classList.toggle("ready", readyCount === allModels.length);
    elements.systemText.textContent = `${readyCount}/${allModels.length} model · ${data.models.device.toUpperCase()}`;
  } catch {
    elements.systemText.textContent = "Sunucuya ulaşılamıyor";
  }
}

elements.dropZone.addEventListener("click", () => elements.fileInput.click());
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") elements.fileInput.click();
});
elements.fileInput.addEventListener("change", () => setFile(elements.fileInput.files[0]));
elements.removeFile.addEventListener("click", clearAll);
elements.clearButton.addEventListener("click", clearAll);
elements.analyzeButton.addEventListener("click", analyze);

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
});

elements.dropZone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
elements.threshold.addEventListener("input", () => {
  elements.thresholdValue.textContent = Number(elements.threshold.value).toFixed(2);
});
document.querySelectorAll('input[name="mode"]').forEach((input) => {
  input.addEventListener("change", applyMode);
});

applyMode();
checkHealth();
