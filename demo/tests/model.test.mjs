import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseDictionary,
  formatTime,
  scoreColor,
  visibleHits,
  clamp,
} from "../static/model.mjs";
test("dictionary formats preserve language and pronunciation variants", () => {
  for (const text of [
    "омепразол\nметформин",
    '["омепразол","метформин"]',
    '{"display":"омепразол"}\n{"name":"метформин"}',
    "name,language\nомепразол,ru\nметформин,ru",
  ])
    assert.equal(parseDictionary(text, "ru").length, 2);
  assert.deepEqual(
    parseDictionary(
      '{"terms":[{"drug":"амоксициллин","aliases":"a;b","spoken_forms":["c"]}]}',
      "ru",
    )[0],
    {
      display: "амоксициллин",
      language: "ru",
      aliases: ["a", "b"],
      spoken_forms: ["c"],
    },
  );
  assert.equal(
    parseDictionary('name,language,aliases\n"A, B",en,"a;b"', "en")[0].display,
    "A, B",
  );
  assert.equal(
    parseDictionary('name,language\n"A""B",en', "en")[0].display,
    'A"B',
  );
  assert.equal(
    parseDictionary('[{"term":"aspirin"}]', "en")[0].display,
    "aspirin",
  );
});
test("invalid imports report actionable errors", () => {
  for (const text of [
    "",
    "[]",
    "[null]",
    "[1]",
    '[{"name":""}]',
    '[{"name":"x","language":"en"}]',
    '{"invalid":true}',
    'name,language\n"unclosed,ru',
  ])
    assert.throws(() => parseDictionary(text, "ru"));
});
test("score colors are bounded; timestamps retain hundredths", () => {
  assert.equal(formatTime(61.234), "1:01.23");
  assert.equal(clamp(2, 0, 1), 1);
  assert.equal(scoreColor(0.9, 0.2, 0.4), scoreColor(1, 0.2, 0.4));
  assert.notEqual(scoreColor(0.2, 0.2, 0.4), scoreColor(0.4, 0.2, 0.4));
  assert.equal(scoreColor(0.2, 0.2, 0.2), "hsl(35 65% 43%)");
});
test("hit views do not mutate results", () => {
  const hits = [
    { start: 2, score: 0.9, tier: "strong" },
    { start: 1, score: 0.3, tier: "possible" },
  ];
  assert.equal(visibleHits(hits, "all", "time")[0].start, 1);
  assert.equal(visibleHits(hits, "strong", "score").length, 1);
  assert.equal(hits[0].start, 2);
});

test("startup prefers a speech encoder and never silently loads the newest MFCC experiment", async () => {
  const { recommendedEncoder, preferredIndex } =
    await import("../static/model.mjs");
  const encoders = [{ id: "phoneme", installed: true, cached: true }];
  assert.equal(recommendedEncoder(encoders, "ru"), "phoneme");
  assert.equal(recommendedEncoder(encoders, "en"), "whisper");
  assert.equal(recommendedEncoder([], "ru"), "whisper");
  assert.equal(
    recommendedEncoder(
      [{ id: "phoneme", installed: false, cached: true }],
      "ru",
    ),
    "whisper",
  );
  assert.equal(
    recommendedEncoder(
      [{ id: "phoneme", installed: true, cached: false }],
      "ru",
    ),
    "whisper",
  );
  const indexes = [
    { language: "ru", encoder: "mfcc" },
    { language: "ru", encoder: "phoneme" },
  ];
  assert.equal(preferredIndex(indexes, "ru", "phoneme"), indexes[1]);
  assert.equal(preferredIndex(indexes, "en", "whisper"), null);
});
test("no-match feedback distinguishes quiet audio, baseline, and rejected scores", async () => {
  const { detectionAdvice } = await import("../static/model.mjs");
  assert.match(
    detectionAdvice({ diagnostics: { quiet_audio: true } }),
    /quiet/,
  );
  assert.match(detectionAdvice({ encoder: "mfcc" }), /speed baseline/);
  assert.match(
    detectionAdvice({
      hits: [],
      possible_threshold: 0.35,
      diagnostics: { best_candidate_score: 0.3 },
    }),
    /0.300.*0.350/,
  );
  assert.match(detectionAdvice({ hits: [] }), /language/);
  assert.match(detectionAdvice({ hits: [{}] }), /estimates/);
});

test("rejected energy candidates are not described as below threshold", async () => {
  const { detectionAdvice } = await import("../static/model.mjs");
  assert.match(
    detectionAdvice({
      hits: [],
      possible_threshold: 0.3,
      diagnostics: { best_candidate_score: 0.5 },
    }),
    /audio energy/,
  );
});

test("no candidates at zero threshold still report no accepted matches", async () => {
  const { detectionAdvice } = await import("../static/model.mjs");
  assert.match(
    detectionAdvice({
      hits: [],
      possible_threshold: 0,
      diagnostics: { best_candidate_score: null },
    }),
    /No accepted matches/,
  );
});

test("dense waveform labels get distinct rows with strong hits first", async () => {
  const { layoutMarkers } = await import("../static/model.mjs");
  const items = Array.from({ length: 7 }, (_, id) => ({
    id,
    center: 100,
    width: 100,
    height: 26,
    priority: id === 4 ? 1 : 0,
  }));
  const layout = layoutMarkers(items, 300);
  assert.equal(layout.placements[4].lane, 0);
  assert.equal(new Set(layout.placements.map((item) => item.lane)).size, 7);
  assert.equal(layout.height, 7 * 34);
});
test("label packing uses actual widths, viewport edges, and reusable gaps", async () => {
  const { layoutMarkers } = await import("../static/model.mjs");
  const items = [
    { id: 0, center: 0, width: 150, height: 20, priority: 0 },
    { id: 1, center: 280, width: 90, height: 30, priority: 0 },
    { id: 2, center: 110, width: 100, height: 20, priority: 0 },
  ];
  for (const width of [300, 180, 90]) {
    const { placements } = layoutMarkers(items, width);
    for (const item of placements)
      assert.ok(
        item.left >= 0 &&
          item.left + Math.min(width, items[item.id].width) <= width,
      );
    for (const a of placements)
      for (const b of placements)
        if (a.id < b.id && a.lane === b.lane)
          assert.ok(
            a.left + Math.min(width, items[a.id].width) + 8 <= b.left ||
              b.left + Math.min(width, items[b.id].width) + 8 <= a.left,
          );
  }
  assert.equal(
    layoutMarkers(items, 300).placements[0].lane,
    layoutMarkers(items, 300).placements[1].lane,
  );
  assert.deepEqual(layoutMarkers([], 300), { placements: [], height: 0 });
});

test("Russian-trained CTC becomes the recommendation only when ready", async () => {
  const { recommendedEncoder } = await import("../static/model.mjs");
  const phoneme = { id: "phoneme", installed: true, cached: true };
  const russian = { id: "russian-ctc", installed: true, cached: true };
  assert.equal(recommendedEncoder([phoneme, russian], "ru"), "russian-ctc");
  assert.equal(
    recommendedEncoder([phoneme, { ...russian, cached: false }], "ru"),
    "phoneme",
  );
  assert.equal(
    recommendedEncoder([phoneme, { ...russian, installed: false }], "ru"),
    "phoneme",
  );
  assert.equal(recommendedEncoder([russian], "en"), "whisper");
});

test("recommended saved indexes preserve the latest dictionary and comparison stays available", async () => {
  const { preferredIndex, comparisonEncoder } =
    await import("../static/model.mjs");
  const terms = [{ display: "нимесил" }];
  const indexes = [
    { language: "ru", encoder: "russian-ctc", terms: [{ display: "old" }] },
    { language: "ru", encoder: "russian-ctc", terms },
  ];
  assert.equal(preferredIndex(indexes, "ru", "russian-ctc", terms), indexes[1]);
  assert.equal(preferredIndex(indexes, "ru", "whisper", terms), null);
  const encoders = [{ id: "russian-ctc", installed: true, cached: true }];
  assert.equal(comparisonEncoder(encoders, "ru", "russian-ctc"), "whisper");
  assert.equal(comparisonEncoder(encoders, "ru", "phoneme"), "russian-ctc");
});
