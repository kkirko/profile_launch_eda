# Position Choice Report (rank 1/2/3)

## Sample
- Cohort: `createdAt` UTC `2026-02-26` — `2026-02-27`
- Sample: только пользователи с LaTeX CV (`cvEnhancedResult`)
- Users total: **521**

## Логика rank
- Из `cvAnalysisResult` берется массив `positioning` (первые 3 элемента).
- Для каждого элемента используется поле `position`.
- `selectedPosition` нормализуется и маппится в номер позиции:
`1` / `2` / `3`.
- Если `selectedPosition` не сопоставляется с одной из 3 позиций, пользователь не попадает в распределение rank.

## Coverage
| group | total_users | with_cvAnalysisResult_nonnull | with_positions3 | with_selected_mapped | share_mapped_% |
|---|---:|---:|---:|---:|---:|
| all | 521 | 521 | 521 | 521 | 100.0 |
| employed | 356 | 356 | 356 | 356 | 100.0 |
| not_employed | 155 | 155 | 155 | 155 | 100.0 |
| exclude_known_companies | 497 | 497 | 497 | 497 | 100.0 |

## Excluded Companies (для группы `exclude_known_companies`)
- `AVO`
- `EPAM`
- `avo в банке`
- `avo интегратор`
- `Сбер`

## Distribution: All
| rank | count | share_% |
|---:|---:|---:|
| 1 | 311 | 59.7 |
| 2 | 96 | 18.4 |
| 3 | 114 | 21.9 |

![Position choice rank all](outputs/figures/position_choice_rank_all.png)

## Distribution: Employed
| rank | count | share_% |
|---:|---:|---:|
| 1 | 212 | 59.6 |
| 2 | 71 | 19.9 |
| 3 | 73 | 20.5 |

![Position choice rank employed](outputs/figures/position_choice_rank_employed.png)

## Distribution: Not Employed
| rank | count | share_% |
|---:|---:|---:|
| 1 | 94 | 60.6 |
| 2 | 23 | 14.8 |
| 3 | 38 | 24.5 |

![Position choice rank not employed](outputs/figures/position_choice_rank_not_employed.png)

## Distribution: Employed + Not Employed (excluding known companies)
| rank | count | share_% |
|---:|---:|---:|
| 1 | 298 | 60.0 |
| 2 | 91 | 18.3 |
| 3 | 108 | 21.7 |

![Position choice rank excluding known companies](outputs/figures/position_choice_rank_exclude_known_companies.png)

## Artifacts
- `outputs/tables/position_choice_coverage.csv`
- `outputs/tables/position_choice_plot_status.csv`
- `outputs/tables/position_choice_excluded_companies.csv`
- `outputs/tables/position_choice_rank_distribution_all.csv`
- `outputs/tables/position_choice_rank_distribution_employed.csv`
- `outputs/tables/position_choice_rank_distribution_not_employed.csv`
- `outputs/tables/position_choice_rank_distribution_exclude_known_companies.csv`

