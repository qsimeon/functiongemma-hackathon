# Fork Benchmark Memo
> Generated: 2026-02-21 16:36
> Forks evaluated: 41 scored, 4 failed

## Top 10

| # | Owner | Score | F1 | Avg Time | On-Device | Notes |
|---|-------|-------|----|----------|-----------|-------|
| 1 | `Rosh2403` | 100.0% | 1.000 | 0ms | 100% | pure rule-based (no model) |
| 2 | `SahilSaxena007` | 100.0% | 1.000 | 0ms | 100% | pure rule-based (no model) |
| 3 | `ishaanvijai` | 91.6% | 1.000 | 270ms | 100% | fast — likely regex + minimal model call |
| 4 | `elena-kalinina` | 89.0% | 1.000 | 343ms | 100% | model-heavy |
| 5 | `avimallick` | 88.5% | 1.000 | 344ms | 100% | model-heavy |
| 6 | `wnwoghd22` | 88.2% | 1.000 | 429ms | 100% | model-heavy |
| 7 | `adi-suresh01` | 87.3% | 0.980 | 357ms | 100% | model-heavy |
| 8 | `vaishnavi2810-code` | 86.6% | 0.970 | 348ms | 100% | model-heavy |
| 9 | `TechCodeBlocks` | 86.4% | 1.000 | 827ms | 100% | model-heavy |
| 10 | `gabikreal1` | 85.6% | 1.000 | 727ms | 100% | model-heavy |

## Interpretation

- **<5ms avg time**: Rule-based only — never calls the model. Fast and accurate on
  known patterns, but may not generalize to unseen phrasings on the hidden leaderboard.
- **<300ms avg time**: Regex-guided + minimal model call. Best balance of accuracy,
  speed, and generalization. **These are the ones worth synthesizing from.**
- **>500ms avg time**: Model-heavy. Accurate but slow — loses time bonus.

## Scoring formula (for reference)

```
Score = 0.20×easy + 0.30×medium + 0.50×hard
level  = 0.60×F1 + 0.15×time_score + 0.25×on_device_ratio
time_score = max(0, 1 − avg_ms / 500)   # full marks if <500ms
```