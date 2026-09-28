export function parseDictionary(text, language) {
  text = text.trim();
  if (!text) throw new Error("The dictionary is empty.");
  let rows;
  if (text.startsWith("[")) rows = JSON.parse(text);
  else if (text.startsWith("{")) {
    try {
      const data = JSON.parse(text);
      rows = Array.isArray(data.terms) ? data.terms : [data];
    } catch {
      rows = text
        .split(/\r?\n/)
        .filter(Boolean)
        .map((line) => JSON.parse(line));
    }
  } else {
    const lines = text.split(/\r?\n/).filter((line) => line.trim());
    const csv = (line) => {
      const cells = [];
      let cell = "",
        quoted = false;
      for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"') {
          if (quoted && line[i + 1] === '"') {
            cell += '"';
            i++;
          } else quoted = !quoted;
        } else if (c === "," && !quoted) {
          cells.push(cell.trim());
          cell = "";
        } else cell += c;
      }
      if (quoted) throw new Error("Unclosed CSV quote.");
      cells.push(cell.trim());
      return cells;
    };
    const header = csv(lines[0]).map((c) => c.toLowerCase());
    if (
      header.includes("language") &&
      header.some((c) => ["display", "name", "drug", "term"].includes(c))
    ) {
      rows = lines
        .slice(1)
        .map((line) =>
          Object.fromEntries(header.map((key, i) => [key, csv(line)[i] || ""])),
        );
    } else rows = lines.map((display) => ({ display }));
  }
  if (!Array.isArray(rows) || !rows.length)
    throw new Error("Provide a list of drug names.");
  const split = (value) =>
    Array.isArray(value)
      ? value
      : typeof value === "string"
        ? value
            .split(";")
            .map((v) => v.trim())
            .filter(Boolean)
        : [];
  return rows.map((row) => {
    if (typeof row === "string") row = { display: row };
    if (!row || typeof row !== "object")
      throw new Error("Each entry needs a name.");
    const display = row.display || row.name || row.drug || row.term;
    if (typeof display !== "string" || !display.trim())
      throw new Error("Each entry needs a name.");
    if (row.language && row.language !== language)
      throw new Error(
        "Choose the language that matches your dictionary. Use one language per index.",
      );
    return {
      display: display.trim(),
      language,
      aliases: split(row.aliases),
      spoken_forms: split(row.spoken_forms),
    };
  });
}
export const formatTime = (seconds) =>
  `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
export const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
export function scoreColor(score, possible, strong) {
  const ratio = clamp(
    (score - possible) / Math.max(0.05, strong - possible),
    0,
    1,
  );
  return `hsl(${Math.round(35 + ratio * 130)} 65% ${Math.round(43 - ratio * 12)}%)`;
}
export function visibleHits(hits, tier, sort) {
  return hits
    .filter((hit) => tier === "all" || hit.tier === tier)
    .sort((a, b) => (sort === "score" ? b.score - a.score : a.start - b.start));
}

// A speed baseline must not silently become the startup choice.
export function recommendedEncoder(encoders, language) {
  const russian = encoders.find((model) => model.id === "russian-ctc");
  if (language === "ru" && russian?.installed && russian.cached)
    return "russian-ctc";
  const phoneme = encoders.find((model) => model.id === "phoneme");
  return language === "ru" && phoneme?.installed && phoneme.cached
    ? "phoneme"
    : "whisper";
}
export function dictionaryKey(terms) {
  return JSON.stringify(
    terms
      .map((term) =>
        JSON.stringify([
          term.display,
          term.aliases || [],
          term.spoken_forms || [],
        ]),
      )
      .sort(),
  );
}
export function comparisonEncoder(encoders, language, current) {
  const recommended = recommendedEncoder(encoders, language);
  return current === recommended && current !== "whisper"
    ? "whisper"
    : recommended;
}
export function preferredIndex(indexes, language, encoder, terms = null) {
  return (
    indexes.find(
      (index) =>
        index.language === language &&
        index.encoder === encoder &&
        (!terms || dictionaryKey(index.terms) === dictionaryKey(terms)),
    ) || null
  );
}
export function detectionAdvice(result) {
  const d = result.diagnostics || {};
  if (d.quiet_audio)
    return "This clip is very quiet. Check microphone input, move closer, and try recording again.";
  if (result.encoder === "mfcc")
    return "MFCC is a speed baseline and often misses words spoken by a different person. Try the recommended speech encoder on this same clip.";
  if (
    !result.hits.length &&
    d.best_candidate_score != null &&
    d.best_candidate_score >= result.possible_threshold
  )
    return "Candidate scores reached the threshold, but no matches passed the audio energy and timestamp checks. Check microphone level and the recording.";
  if (!result.hits.length && d.best_candidate_score != null)
    return `Best candidate score ${d.best_candidate_score.toFixed(3)} is below the possible threshold ${result.possible_threshold.toFixed(3)}. Try another encoder or add the pronunciation you use. Lower thresholds can also produce false matches.`;
  if (!result.hits.length)
    return "No accepted matches. Check the dictionary language and names, and try a different encoder or pronunciation.";
  return "Matches are estimates. Listen to each marked span to check the term.";
}

// Pack rendered label rectangles, rather than audio spans, into as many rows as needed.
export function layoutMarkers(items, width, gap = 8) {
  const lanes = [];
  const rowHeight = Math.max(0, ...items.map((item) => item.height)) + gap;
  const placements = new Map();
  [...items]
    .sort(
      (a, b) => b.priority - a.priority || a.center - b.center || a.id - b.id,
    )
    .forEach((item) => {
      const labelWidth = Math.min(item.width, width);
      const left = clamp(
        item.center - labelWidth / 2,
        0,
        Math.max(0, width - labelWidth),
      );
      const right = left + labelWidth;
      let lane = lanes.findIndex((intervals) =>
        intervals.every(
          (interval) =>
            right + gap <= interval.left || left >= interval.right + gap,
        ),
      );
      if (lane < 0) {
        lane = lanes.length;
        lanes.push([]);
      }
      lanes[lane].push({ left, right });
      placements.set(item.id, {
        id: item.id,
        left,
        top: lane * rowHeight,
        lane,
      });
    });
  return {
    placements: items.map((item) => placements.get(item.id)),
    height: lanes.length * rowHeight,
  };
}
