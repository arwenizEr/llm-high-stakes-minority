# Downstream-Aware Prompt Compressor — Results

Model claude-haiku-4-5, HotpotQA distractor. Granularity sentence. Features: ['intrinsic_ppl', 'position', 'n_words', 'q_overlap'].

## Span classifier (downstream label ~ features)

- held-out accuracy: **0.780**
- **load-bearing recall: 0.071** (the spans we must not drop)
- class balance (load-bearing frac): 0.10 — imbalanced
- confusion ['droppable(0)', 'load_bearing(1)']: [[295, 45], [39, 3]]

## Pareto: downstream-aware vs LLMLingua-2 (held-out test)

| rate | downstream acc | downstream tokens | LLMLingua-2 acc | LLMLingua-2 tokens |
|---|---|---|---|---|
| 0 (full) | 0.767 | 1570 | — | — |
| 0.2 | 0.683 | 1261 | 0.750 | 1293 |
| 0.3 | 0.650 | 1124 | 0.733 | 1135 |
| 0.5 | 0.550 | 843 | 0.683 | 845 |

## Headline (accuracy at matched token budget)

- rate 0.2: downstream 0.683 @ 1261 tok vs LLMLingua-2 0.747 at the same cost = **-0.063**
- rate 0.3: downstream 0.650 @ 1124 tok vs LLMLingua-2 0.732 at the same cost = **-0.082**
- rate 0.5: downstream 0.550 @ 843 tok vs LLMLingua-2 0.683 at the same cost = **-0.133**

## Verdict

**No / inconclusive.** Mean accuracy gain vs LLMLingua-2 at matched tokens is **-0.093** (0/3 rates above). Downstream labels do not beat intrinsic perplexity at this scale. Likely causes: small label set (730+382 spans), coarse granularity, or the load-bearing recall (0.07) too low to protect the spans that matter. A clean negative is still a result.