import {
  parseDictionary,
  formatTime,
  clamp,
  scoreColor,
  visibleHits,
  recommendedEncoder,
  preferredIndex,
  detectionAdvice,
  layoutMarkers,
  comparisonEncoder,
} from "./model.mjs";
const $ = (id) => document.getElementById(id);
let config,
  terms = [],
  indexes = [],
  active = null,
  busy = false,
  result = null,
  selected = null,
  tier = "all";
let recorder,
  stream,
  audioContext,
  analyser,
  animation,
  timer,
  recordStart,
  chunks = [],
  audioUrl,
  lastAudio,
  stopAt;
const element = (tag, text, className) => {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
};
function error(message) {
  $("error").textContent = message || "";
  $("error").hidden = !message;
}
async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      data.error || "The local server could not complete this request.",
    );
  return data;
}
function guard(task) {
  return async (...args) => {
    error("");
    try {
      await task(...args);
    } catch (e) {
      error(e.message);
    }
  };
}
function controls() {
  $("build").disabled = busy || !!recorder || !terms.length;
  $("record").disabled = busy || !active;
  $("audio-file").disabled = busy || !!recorder || !active;
  $("upload-label").classList.toggle("disabled", $("audio-file").disabled);
  for (const el of document.querySelectorAll(
    ".dictionary input,.dictionary select,.dictionary textarea,.dictionary button,.saved-label select",
  ))
    el.disabled = busy || !!recorder;
  $("build").disabled = busy || !!recorder || !terms.length;
  $("ready-chip").textContent = active
    ? `${active.terms.length} terms · ready`
    : "No index yet";
  $("ready-chip").classList.toggle("ready", !!active);
  $("reanalyze").hidden = !lastAudio || !active;
  $("reanalyze").disabled = busy || !!recorder;
  $("try-encoder").disabled = busy || !!recorder;
  $("active-warning").hidden = active?.encoder !== "mfcc";
  $("active-warning").textContent =
    "MFCC is an experimental speed baseline. It missed terms in our human recordings. Use Whisper or phoneme for this demo.";
  if (!recorder)
    $("record-subtitle").textContent = active
      ? `${active.name} · ${active.language === "ru" ? "Russian" : "English"} · ${active.encoder}`
      : "Build or select an index, then press Record.";
}
function invalidate() {
  active = null;
  $("saved-index").value = "";
  controls();
}
function renderTerms() {
  $("terms").replaceChildren();
  terms.forEach((term, i) => {
    const row = element("div", undefined, "term-row");
    const name = element("input");
    name.value = term.display;
    name.placeholder = "Drug or keyword";
    name.setAttribute("aria-label", `Term ${i + 1}`);
    name.addEventListener("input", () => {
      term.display = name.value;
      invalidate();
    });
    const aliases = element("input", undefined, "aliases");
    aliases.value = [
      ...(term.aliases || []),
      ...(term.spoken_forms || []),
    ].join("; ");
    aliases.placeholder = "Other pronunciations / aliases (optional)";
    aliases.setAttribute("aria-label", `Aliases for term ${i + 1}`);
    aliases.addEventListener("input", () => {
      term.aliases = aliases.value
        .split(";")
        .map((v) => v.trim())
        .filter(Boolean);
      term.spoken_forms = [];
      invalidate();
    });
    const remove = element("button", "×");
    remove.setAttribute("aria-label", `Remove ${term.display || "term"}`);
    remove.addEventListener("click", () => {
      terms.splice(i, 1);
      invalidate();
      renderTerms();
    });
    row.append(name, aliases, remove);
    $("terms").append(row);
  });
  $("term-count").textContent = `(${terms.length})`;
  controls();
}
function defaults() {
  const [strong, possible] =
    config.thresholds[$("language").value][$("encoder").value];
  $("strong").value = strong;
  $("possible").value = possible;
}
function modelInfo() {
  const model = config.encoders.find((v) => v.id === $("encoder").value);
  $("encoder-hint").textContent = model.hint;
  $("lookup-hint").textContent = {
    dtw: "Lookup: subsequence dynamic time warping. Pronunciations: local Piper voice.",
    phoneme:
      "Lookup: phoneme sequence matching. Dictionary pronunciations: local eSpeak; Piper is not needed.",
    characters:
      "Lookup: Russian CTC character sequence matching. Dictionary tokens come directly from spelling; no TTS or transcript.",
  }[model.lookup];
  for (const option of $("encoder").options) {
    const definition = config.encoders.find((item) => item.id === option.value);
    option.disabled =
      definition.languages &&
      !definition.languages.includes($("language").value);
  }
  const ready =
    model.installed &&
    model.cached &&
    (!model.needs_voice || config.voice_cached[$("language").value]);
  $("readiness").classList.toggle("missing", !ready);
  $("readiness").textContent = !model.installed
    ? "Optional dependencies missing. See demo/README.md to install this encoder."
    : ready
      ? "✓ Models ready for offline use"
      : "Model or voice not cached. Enable downloads below, or prepare models first.";
}
function refreshIndexes() {
  $("saved-index").replaceChildren(new Option("Choose a saved index…", ""));
  indexes.forEach((v) =>
    $("saved-index").append(
      new Option(
        `${v.name} · ${v.language.toUpperCase()} · ${v.encoder}`,
        v.id,
      ),
    ),
  );
  $("saved-index").value = active?.id || "";
}
function chooseIndex(info) {
  active = info;
  terms = structuredClone(info.terms);
  $("name").value = info.name;
  $("language").value = info.language;
  $("encoder").value = info.encoder;
  $("strong").value = info.strong_threshold;
  $("possible").value = info.possible_threshold;
  $("top-k").value = info.top_k;
  $("downloads").checked = !info.offline;
  modelInfo();
  renderTerms();
  refreshIndexes();
}
async function poll(id, prefix) {
  const box = $(`${prefix}-progress`);
  box.hidden = false;
  try {
    while (true) {
      const job = await request(`/api/jobs/${id}`);
      $(`${prefix}-message`).textContent = job.message;
      $(`${prefix}-percent`).textContent = `${job.progress}%`;
      $(`${prefix}-bar`).value = job.progress;
      if (job.state === "error") throw new Error(job.message);
      if (job.state === "complete") return job.result;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  } finally {
    box.hidden = true;
  }
}
async function buildIndex(payload) {
  const { job_id } = await request("/api/indexes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const info = await poll(job_id, "index");
  indexes.unshift(info);
  chooseIndex(info);
  config = await request("/api/config");
  modelInfo();
  return info;
}
$("build").addEventListener(
  "click",
  guard(async () => {
    busy = true;
    controls();
    try {
      await buildIndex({
        name: $("name").value,
        language: $("language").value,
        terms,
        encoder: $("encoder").value,
        top_k: Number($("top-k").value),
        strong_threshold: Number($("strong").value),
        possible_threshold: Number($("possible").value),
        allow_downloads: $("downloads").checked,
      });
    } finally {
      busy = false;
      controls();
    }
  }),
);
$("reanalyze").addEventListener(
  "click",
  guard(() => analyze(lastAudio)),
);
$("try-encoder").addEventListener(
  "click",
  guard(async () => {
    const original = indexes.find((index) => index.id === result.index_id);
    const model = comparisonEncoder(
      config.encoders,
      result.language,
      result.encoder,
    );
    const [strong, possible] = config.thresholds[result.language][model];
    busy = true;
    controls();
    try {
      await buildIndex({
        name: `${original.name.slice(0, 60)} / ${model}`,
        language: original.language,
        terms: original.terms,
        encoder: model,
        top_k: 3,
        strong_threshold: strong,
        possible_threshold: possible,
        allow_downloads: $("downloads").checked,
      });
      await analyze(lastAudio);
    } finally {
      busy = false;
      controls();
    }
  }),
);
$("saved-index").addEventListener("change", () => {
  const info = indexes.find((v) => v.id === $("saved-index").value);
  if (info) chooseIndex(info);
  else invalidate();
});
$("name").addEventListener("input", invalidate);
$("language").addEventListener("change", () => {
  terms = [];
  $("encoder").value = recommendedEncoder(config.encoders, $("language").value);
  invalidate();
  defaults();
  modelInfo();
  renderTerms();
});
$("encoder").addEventListener("change", () => {
  invalidate();
  defaults();
  modelInfo();
});
for (const id of ["strong", "possible", "top-k", "downloads"])
  $(id).addEventListener("change", invalidate);
$("sample").addEventListener("click", () => {
  terms = structuredClone(
    config.samples.filter((v) => v.language === $("language").value),
  );
  $("name").value =
    `${$("language").value === "ru" ? "Russian" : "English"} pharma`;
  invalidate();
  renderTerms();
});
$("add-term").addEventListener("click", () => {
  if (terms.length >= 100) {
    error("The demo supports up to 100 terms per dictionary.");
    return;
  }
  terms.push({
    display: "",
    language: $("language").value,
    aliases: [],
    spoken_forms: [],
  });
  invalidate();
  renderTerms();
  $("terms").lastElementChild.querySelector("input").focus();
});
$("import-toggle").addEventListener("click", () => {
  $("import-panel").hidden = !$("import-panel").hidden;
});
function applyImport(text) {
  terms = parseDictionary(text, $("language").value);
  invalidate();
  renderTerms();
  $("import-panel").hidden = true;
}
$("import-apply").addEventListener(
  "click",
  guard(() => applyImport($("import-text").value)),
);
$("dictionary-file").addEventListener(
  "change",
  guard(async (event) => {
    const file = event.target.files[0];
    if (file) applyImport(await file.text());
    event.target.value = "";
  }),
);
async function analyze(blob) {
  busy = true;
  controls();
  error("");
  try {
    const { job_id } = await request(`/api/runs?index_id=${active.id}`, {
      method: "POST",
      body: blob,
    });
    const data = await poll(job_id, "run");
    const response = await fetch(data.audio_url);
    if (!response.ok) throw new Error("Could not load playback audio.");
    const audio = await response.blob();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(audio);
    $("playback").src = audioUrl;
    lastAudio = blob;
    result = data;
    tier = "all";
    $("filters")
      .querySelectorAll("button")
      .forEach((button) =>
        button.classList.toggle("active", button.dataset.tier === "all"),
      );
    selected = null;
    stopAt = null;
    showResults();
  } finally {
    busy = false;
    controls();
  }
}
$("audio-file").addEventListener(
  "change",
  guard(async (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (file) await analyze(file);
  }),
);
function cleanupRecording() {
  clearInterval(timer);
  cancelAnimationFrame(animation);
  stream?.getTracks().forEach((track) => track.stop());
  audioContext?.close();
  stream = null;
  audioContext = null;
  recorder = null;
  $("record-zone").classList.remove("is-recording");
  $("record").replaceChildren(
    element("span", "●"),
    document.createTextNode(" Record audio"),
  );
  $("record-title").textContent = "Your next recording starts here";
  $("live-wave").hidden = true;
  controls();
}
function liveWave() {
  const canvas = $("live-wave"),
    ctx = canvas.getContext("2d");
  const samples = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(samples);
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = 44 * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, canvas.clientWidth, 44);
  ctx.beginPath();
  ctx.strokeStyle = "#b05b42";
  ctx.lineWidth = 1.5;
  samples.forEach((v, i) => {
    const x = (i / samples.length) * canvas.clientWidth,
      y = (v / 255) * 44;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  animation = requestAnimationFrame(liveWave);
}
$("record").addEventListener(
  "click",
  guard(async () => {
    if (recorder) {
      if (recorder.state !== "inactive") recorder.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder)
      throw new Error(
        "Microphone recording is unavailable in this browser. Use audio upload.",
      );
    busy = true;
    controls();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = [
        "audio/webm;codecs=opus",
        "audio/mp4",
        "audio/ogg;codecs=opus",
      ].find((value) => MediaRecorder.isTypeSupported(value));
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
      chunks = [];
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size) chunks.push(event.data);
      });
      recorder.addEventListener(
        "stop",
        guard(async () => {
          const blob = new Blob(chunks, { type: recorder.mimeType });
          cleanupRecording();
          await analyze(blob);
        }),
      );
      recorder.addEventListener("error", () => {
        cleanupRecording();
        error("Recording failed. Try another browser or upload a clip.");
      });
      audioContext = new AudioContext();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      audioContext.createMediaStreamSource(stream).connect(analyser);
      recorder.start(250);
      busy = false;
      $("playback").pause();
      recordStart = performance.now();
      $("record-zone").classList.add("is-recording");
      $("record-title").textContent = "Listening…";
      $("record").textContent = "■ Stop & detect";
      $("live-wave").hidden = false;
      controls();
      liveWave();
      timer = setInterval(() => {
        const seconds = (performance.now() - recordStart) / 1000;
        $("record-subtitle").textContent =
          `${formatTime(seconds)} / 2:00 · Speak your dictionary terms`;
        if (seconds >= config.max_duration && recorder?.state === "recording")
          recorder.stop();
      }, 200);
    } catch (e) {
      busy = false;
      cleanupRecording();
      throw new Error(
        e.name === "NotAllowedError"
          ? "Microphone permission was denied. Allow access in your browser, or upload an audio file."
          : e.message,
      );
    }
  }),
);
function hitColor(hit) {
  return scoreColor(
    hit.score,
    result.possible_threshold,
    result.strong_threshold,
  );
}
function seekHit(hit) {
  selected = hit;
  $("playback").currentTime = Math.max(0, hit.start - 0.12);
  stopAt = Math.min(result.duration, hit.end + 0.2);
  $("playback")
    .play()
    .catch((e) => error(e.message));
  renderHits();
  renderMarkers();
}
function renderMarkers() {
  const container = $("wave-container"),
    width = container.clientWidth;
  $("markers").replaceChildren();
  const items = result.hits.map((hit, id) => {
    const color = hitColor(hit),
      span = element("div", undefined, "hit-span");
    span.style.cssText = `left:${(hit.start / result.duration) * 100}%;width:${((hit.end - hit.start) / result.duration) * 100}%;--hit-color:${color}`;
    const marker = element(
      "button",
      hit.term,
      `marker${hit === selected ? " selected" : ""}`,
    );
    marker.style.setProperty("--hit-color", color);
    marker.title = `${hit.term} · ${formatTime(hit.start)}–${formatTime(hit.end)} · ${hit.tier} · score ${hit.score.toFixed(3)}`;
    marker.setAttribute("aria-label", `Play ${marker.title}`);
    marker.addEventListener("click", (event) => {
      event.stopPropagation();
      seekHit(hit);
    });
    $("markers").append(span, marker);
    const bounds = marker.getBoundingClientRect();
    return {
      id,
      marker,
      center: ((hit.start + hit.end) / 2 / result.duration) * width,
      width: bounds.width,
      height: bounds.height,
      priority: hit.tier === "strong" ? 1 : 0,
    };
  });
  const layout = layoutMarkers(items, width);
  const waveTop = Math.max(16, layout.height + 8);
  container.style.setProperty("--wave-top", `${waveTop}px`);
  container.style.height = `${waveTop + 90}px`;
  layout.placements.forEach(({ id, left, top }) => {
    const item = items[id];
    item.marker.style.left = `${left}px`;
    item.marker.style.top = `${top}px`;
  });
  drawWave();
}
function renderHits() {
  const hits = visibleHits(result.hits, tier, $("sort").value);
  $("hits").replaceChildren();
  if (!hits.length)
    $("hits").append(
      element(
        "p",
        result.hits.length
          ? "No matches in this tier."
          : "No dictionary terms detected above the current thresholds.",
        "no-hits",
      ),
    );
  hits.forEach((hit) => {
    const button = element(
      "button",
      undefined,
      `hit-card${hit === selected ? " selected" : ""}`,
    );
    button.style.setProperty("--hit-color", hitColor(hit));
    const detail = element("div");
    detail.append(
      element("span", hit.term, "hit-name"),
      element(
        "span",
        `${formatTime(hit.start)} – ${formatTime(hit.end)}`,
        "hit-time",
      ),
    );
    const score = element("div", undefined, "hit-score");
    score.append(
      element("b", hit.score.toFixed(3)),
      element("small", `${hit.tier} · match score`),
    );
    button.append(element("span", "▶", "hit-play"), detail, score);
    button.addEventListener("click", () => seekHit(hit));
    $("hits").append(button);
  });
}
function drawWave() {
  if (!result) return;
  const canvas = $("waveform"),
    width = canvas.clientWidth,
    height = canvas.clientHeight;
  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#639078";
  ctx.lineWidth = Math.max(1, (width / result.peaks.length) * 0.8);
  ctx.beginPath();
  const center =
    parseFloat($("wave-container").style.getPropertyValue("--wave-top")) + 36;
  result.peaks.forEach(([low, high], i) => {
    const x = (i / result.peaks.length) * width;
    ctx.moveTo(x, center + low * 35);
    ctx.lineTo(x, center + high * 35);
  });
  ctx.stroke();
}
function showResults() {
  $("detection-advice").textContent = detectionAdvice(result);
  const alternative = comparisonEncoder(
    config.encoders,
    result.language,
    result.encoder,
  );
  $("try-encoder").hidden = alternative === result.encoder;
  $("try-encoder").textContent =
    `Try ${config.encoders.find((model) => model.id === alternative).name} on this clip`;
  $("results").hidden = false;
  $("empty-results").hidden = true;
  $("result-meta").replaceChildren();
  for (const text of [
    result.dictionary_name,
    `${result.duration.toFixed(2)} seconds`,
    `${result.runtime_seconds.toFixed(2)}s processing`,
    `${result.encoder} / ${result.lookup}`,
  ])
    $("result-meta").append(element("span", text));
  $("all-count").textContent = result.hits.length;
  $("strong-count").textContent = result.hits.filter(
    (v) => v.tier === "strong",
  ).length;
  $("possible-count").textContent = result.hits.filter(
    (v) => v.tier === "possible",
  ).length;
  $("time-axis").replaceChildren(
    ...[0, 0.25, 0.5, 0.75, 1].map((v) =>
      element("span", formatTime(result.duration * v)),
    ),
  );
  $("wave-container").setAttribute("aria-valuemax", result.duration);
  renderMarkers();
  renderHits();
  drawWave();
  $("playhead").style.left = "0%";
}
$("filters").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || !result) return;
  tier = button.dataset.tier;
  $("filters")
    .querySelectorAll("button")
    .forEach((v) => v.classList.toggle("active", v === button));
  renderHits();
});
$("sort").addEventListener("change", () => result && renderHits());
$("playback").addEventListener("timeupdate", () => {
  if (!result) return;
  const seconds = $("playback").currentTime;
  $("playhead").style.left = `${(seconds / result.duration) * 100}%`;
  $("wave-container").setAttribute("aria-valuenow", seconds.toFixed(2));
  if (stopAt !== null && seconds >= stopAt) {
    $("playback").pause();
    stopAt = null;
  }
});
$("wave-container").addEventListener("click", (event) => {
  if (!result) return;
  const rect = $("wave-container").getBoundingClientRect();
  stopAt = null;
  $("playback").currentTime =
    clamp((event.clientX - rect.left) / rect.width, 0, 1) * result.duration;
});
$("wave-container").addEventListener("keydown", (event) => {
  if (!result) return;
  if (["ArrowRight", "ArrowLeft", " "].includes(event.key)) {
    event.preventDefault();
    stopAt = null;
    if (event.key === " ") {
      if ($("playback").paused)
        $("playback")
          .play()
          .catch((e) => error(e.message));
      else $("playback").pause();
    } else
      $("playback").currentTime = clamp(
        $("playback").currentTime + (event.key === "ArrowRight" ? 1 : -1),
        0,
        result.duration,
      );
  }
});
$("download").addEventListener("click", () => {
  const { peaks, audio_url, ...data } = result;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const link = element("a");
  link.href = url;
  link.download = "keyword-results.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
let lastWaveWidth = 0;
new ResizeObserver((entries) => {
  const width = entries[0].contentRect.width;
  if (result && width !== lastWaveWidth) {
    lastWaveWidth = width;
    renderMarkers();
  } else drawWave();
}).observe($("waveform"));
window.addEventListener("beforeunload", () => {
  stream?.getTracks().forEach((track) => track.stop());
  if (audioUrl) URL.revokeObjectURL(audioUrl);
});
guard(async () => {
  config = await request("/api/config");
  indexes = config.indexes;
  config.encoders.forEach((v) => $("encoder").append(new Option(v.name, v.id)));
  $("encoder").value = recommendedEncoder(config.encoders, "ru");
  const draft = indexes.find((index) => index.language === "ru");
  terms = structuredClone(
    draft ? draft.terms : config.samples.filter((v) => v.language === "ru"),
  );
  if (draft) $("name").value = draft.name;
  defaults();
  modelInfo();
  renderTerms();
  refreshIndexes();
  const preferred = preferredIndex(indexes, "ru", $("encoder").value, terms);
  if (preferred) chooseIndex(preferred);
})();
