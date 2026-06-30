# Phase 2.5 — Stress Test Results (frozen pre-reg)

Date 2026-06-29. cheap=claude-haiku-4-5, frontier=claude-sonnet-4-6.

## H1 — Replication at larger n
- confident-error set n = **28**, self-verify caught **6** => recall **0.21** (95% CI 0.10–0.40)
- threshold: n>=30, recall>=0.30, CI lower>0 -> **FAIL**

## H2 — Asymmetry vs leakage (decisive)
- format-checkable CE: recall 0.13333333333333333 (2/15)
- **non-format-checkable CE: recall 0.3076923076923077 (4/13), CI 0.13–0.58**
- threshold: non-format recall>0, CI lower>0, within 0.15 of format -> **PASS — asymmetry real**

## H3 — Re-anchored g + rule check (per workload, combined = k3 selfcons + verify)
| workload | n | g | C_w | C_w_max | predicted | realized | match |
|---|---|---|---|---|---|---|---|
| gsm8k | 120 | 0.59 | 1.10 | 0.00 | NO | NO | OK |
| arc_challenge | 120 | 0.65 | 1.15 | 0.00 | NO | NO | OK |
| squad | 120 | 0.72 | 1.00 | 0.00 | NO | NO | OK |
| math | 200 | 1.52 | 2.65 | 1.75 | NO | NO | OK |

- rule predicts realized GO/NO-GO on all workloads: **PASS** (every verdict stated at its g under 2026-06-29 prices).

## Phase-3 go/no-go

**HOLD — do not advance to Phase 3 as-is.** Failed: H1. H1 FAIL => the breakthrough does not replicate at n; revert to suggestive. 