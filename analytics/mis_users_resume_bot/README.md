# MIS: Users Resume Bot (Candidate Analytics)

## 1) Summary
- Cohort filter (createdAt UTC): **26-27 Feb 2026**
- Users total: **723**
- Coverage `cvEnhancedResult`: **72.1%**
- Coverage `talentCard.jobs`: **72.1%**
- Coverage `talentCard.overall_skills`: **72.1%**
- Coverage `talentCard.specialist_category`: **72.1%**

Top-3 domains (excluding Other/Not specified):
| domain                     |   count |
|:---------------------------|--------:|
| Product/Project Management |     236 |
| Finance/Legal/HR           |     138 |
| Engineering/IT             |      61 |

Top-3 regions (excluding Other/Not specified):
| region         |   count |
|:---------------|--------:|
| Москва         |     124 |
| Russia         |      27 |
| Minsk, Belarus |      26 |

Top-3 companies (excluding Other/Not specified):
| company      |   count |
|:-------------|--------:|
| Сбер         |       7 |
| EPAM Systems |       4 |
| Lyft         |       3 |

### Key observations
- Срез MIS ограничен cohort-фильтром по `createdAt` (UTC): 2026-02-26 to 2026-02-27.
- База содержит 723 профилей.
- Покрытие `cvEnhancedResult`: 72.1%.
- Покрытие `talentCard.jobs`: 72.1%.
- Покрытие `talentCard.overall_skills`: 72.1%.
- Покрытие `talentCard.specialist_category`: 72.1%.
- LaTeX-парсинг нашел `ExpHeader` у 71.5% пользователей.
- LaTeX skills section найдена у 72.1% пользователей.
- Гео-мэппинг схлопнул варианты Russia/Россия/РФ: 27 записей перешли в канон `Russia`.
- Industry анализируется только на subset с заполненным industry: 15.5% пользователей.
- `region=Not specified` остается у 33.1% базы; `company=Not specified` — у 28.4%.
- В fallback цепочку региона добавлено альтернативных geo-колонок: 0.
- CV language среди пользователей с cvEnhancedResult: en: 52.6%, ru: 47.4%; no_latex=202 (27.9%).
- Топ tools: Jira, Confluence, Sql, Miro, Figma.
- Статус занятости: employed 49.2%, not_employed 21.4%, unknown 29.3%.
- Unknown после улучшения парсинга периодов: 212 (29.3%), до улучшения: 212 (29.3%), rescued periods: 0.

## 2) Coverage / Parsing validation
| metric                               |   value |
|:-------------------------------------|--------:|
| users_total                          |   723   |
| coverage_cvEnhancedResult_%          |    72.1 |
| coverage_talentCard_jobs_%           |    72.1 |
| coverage_overall_skills_%            |    72.1 |
| coverage_specialist_category_%       |    72.1 |
| coverage_industry_talentCard_%       |    15.5 |
| users_with_latex_block               |   521   |
| share_users_with_latex_block_%       |    72.1 |
| users_with_expheader                 |   517   |
| share_users_with_expheader_%         |    71.5 |
| users_with_skills_section            |   521   |
| share_users_with_skills_section_%    |    72.1 |
| users_with_languages_section         |   418   |
| share_users_with_languages_section_% |    57.8 |
| users_with_education_section         |    38   |
| share_users_with_education_section_% |     5.3 |
| company_comparable_users             |   512   |
| current_company_matches              |   487   |
| current_company_match_rate_%         |    95.1 |

Columns inventory (real CSV structure + non-null profile + length stats):
- `outputs/tables/columns_inventory.csv`
- `outputs/tables/createdAt_date_counts_all.csv`
- `outputs/tables/createdAt_date_counts_filtered.csv`

## 3) Domains & Geography
![Top domains](outputs/figures/02_top_domains.png)
![Top regions](outputs/figures/03_top_regions.png)
![Domain x Region heatmap (row share)](outputs/figures/06_heatmap_domain_region_share.png)
![Region composition by domains (100% stacked)](outputs/figures/07_stacked_region_domain_top10.png)

## 4) Companies & Seniority
![Top companies](outputs/figures/04_top_companies.png)
![Top industries (subset)](outputs/figures/05_top_industries_subset.png)
![Seniority x Company (100% stacked)](outputs/figures/08_stacked_seniority_company_top15.png)
![Domain x Company (100% stacked)](outputs/figures/09_stacked_domain_company_top15.png)

## 5) Skills & Stack
![Top tools](outputs/figures/10_top_tools.png)
![Top skills](outputs/figures/11_top_skills.png)
![Domain x Tools heatmap (row share)](outputs/figures/12_heatmap_domain_tools_share.png)

## 6) Стратификация выборки
CV language (among users with cvEnhancedResult): `ru/en`; `no_latex` учитывается только как покрытие.
- no_latex_count: **202** (27.9% базы)
<p><img src="outputs/figures/13_donut_seniority_filled.png" width="49%"><img src="outputs/figures/14_donut_cv_generation_language.png" width="49%"></p>

Ключевые таблицы стратификации:
- `outputs/tables/strata_top20.csv`
- `outputs/tables/domain_distribution.csv`
- `outputs/tables/role_family_distribution.csv`
- `outputs/tables/seniority_distribution.csv`
- `outputs/tables/experience_bin_distribution.csv`
- `outputs/tables/leadership_distribution.csv`
- `outputs/tables/cv_generation_language_distribution.csv`
- `outputs/tables/cv_language_coverage.csv`
- `outputs/tables/language_audit.csv`

## 7) Employment status (working vs not working)
| employment_status   |   count |   share_% |
|:--------------------|--------:|----------:|
| employed            |     356 |      49.2 |
| not_employed        |     155 |      21.4 |
| unknown             |     212 |      29.3 |

Ключевые наблюдения:
- Домен с максимальной долей employed: `Data/ML/Analytics` (85.7%).
- Домен с максимальной долей not_employed: `Marketing/Sales` (50.0%).
- Регион с максимальной долей not_employed: `Serbia` (66.7%).
- Сеньорность с максимальной долей not_employed: `C-level` (35.6%).

![Employment status overall](outputs/figures/16_employment_status_overall.png)
![Employment status by domain (100% stacked)](outputs/figures/17_employment_status_by_domain.png)
![Months since last end-date (not employed)](outputs/figures/18_months_since_last_end_hist.png)
![Not employed: top last companies](outputs/figures/19_not_employed_top_last_companies.png)
![Not employed: top last titles](outputs/figures/20_not_employed_top_last_titles.png)
![Not employed: historical top companies](outputs/figures/21_not_employed_history_top_companies.png)

Таблицы employment status:
- `outputs/tables/employment_status_summary.csv`
- `outputs/tables/employment_status_by_domain.csv`
- `outputs/tables/employment_status_by_region.csv`
- `outputs/tables/employment_status_by_seniority.csv`
- `outputs/tables/not_employed_top_last_companies.csv`
- `outputs/tables/not_employed_top_last_titles.csv`
- `outputs/tables/not_employed_months_since_last_end.csv`
- `outputs/tables/not_employed_history_top_companies.csv`
- `outputs/tables/not_employed_history_top_titles.csv`

### Unknown deep dive
- unknown_count: **212** (29.3% of cohort); before parser improvements: **212** (29.3%).
- Top unknown reasons:
  - `no_jobs_any`: 202 (95.3%)
  - `other`: 5 (2.4%)
  - `period_present_but_all_parse_failed`: 5 (2.4%)
- Unknown users without LaTeX: **95.3%**.
- Unknown users with jobs but no period: **0.0%**.
- Rescued failed period rows by parser upgrade: **0**.
Links:
- `outputs/tables/employment_unknown_breakdown.csv`
- `outputs/tables/employment_unknown_parse_failed_periods_top.csv`
- `outputs/tables/employment_unknown_users.csv`
- `outputs/tables/employment_unknown_crosstab_sources.csv`
- `outputs/tables/employment_unknown_before_after.csv`
- `outputs/tables/employment_unknown_reason_shift.csv`

## 8) Not specified research
| field     |   total_missing_count |   share_missing_% |   share_filled_by_fallback | source_breakdown                                                                            |
|:----------|----------------------:|------------------:|---------------------------:|:--------------------------------------------------------------------------------------------|
| domain    |                     0 |               0   |                       98.5 | inferred:98.5%; talentCard:1.5%                                                             |
| region    |                   239 |              33.1 |                       20.7 | latex_expheader:46.2%; not_specified:33.1%; latex_header:14.0%; talentCard:6.8%             |
| company   |                   202 |              27.9 |                        0.6 | latex_expheader:71.5%; not_specified:27.9%; talentCard:0.6%                                 |
| seniority |                   292 |              40.4 |                       30.8 | not_specified:40.4%; talentCard:28.8%; inferred_job_title:16.7%; inferred_header_role:14.1% |
| industry  |                   611 |              84.5 |                        0   | not_specified:84.5%; talentCard:15.5%                                                       |

Пустоты уменьшались по fallback-цепочкам:
- `region_filled`: `latex_expheader -> talentCard -> latex_header -> alt_geo_columns`
- `seniority_filled`: `talentCard -> inferred_job_title -> inferred_header_role`
- `domain_filled`: `talentCard.specialist_category -> inferred role family`

Таблицы исследования Not specified:
- `outputs/tables/not_specified_deep_dive_summary.csv`
- `outputs/tables/not_specified_deep_dive_region_not_specified_breakdown.csv`
- `outputs/tables/not_specified_deep_dive_region_not_specified_domain.csv`
- `outputs/tables/not_specified_deep_dive_region_not_specified_seniority.csv`
- `outputs/tables/not_specified_deep_dive_region_alt_columns.csv`
- `outputs/tables/not_specified_deep_dive_company_not_specified_breakdown.csv`
- `outputs/tables/not_specified_deep_dive_company_not_specified_job_titles.csv`
- `outputs/tables/not_specified_deep_dive_company_not_specified_region.csv`

## 9) Domain Other
Исследование домена `Other` вынесено в отдельный отчёт: `REPORT_OTHERS.md`.

## 10) Appendix
Артефакты:
- Figures: `outputs/figures/*.png`
- Tables: `outputs/tables/*.csv`
- Other report: `REPORT_OTHERS.md` + `outputs/others/*`
- Notebook: `notebooks/mis_users_resume_bot.ipynb`
- Geo mapping audit: `outputs/tables/geo_mapping_audit.csv`
- Geo mapping top-50: `outputs/tables/geo_mapping_top50.csv`
- Company mapping collisions: `outputs/tables/company_mapping_collisions.csv`

How to reproduce:
```bash
python analytics/mis_users_resume_bot/src/build_mis.py \
  --input /mnt/data/prointerview-prod.users.csv \
  --base-dir analytics/mis_users_resume_bot
```
