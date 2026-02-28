# MIS: Users Resume Bot (Candidate Analytics)

## 1) Summary
- Users total: **730**
- Coverage `cvEnhancedResult`: **72.2%**
- Coverage `talentCard.jobs`: **72.2%**
- Coverage `talentCard.overall_skills`: **72.2%**
- Coverage `talentCard.specialist_category`: **71.4%**

Top-3 domains (excluding Other/Not specified):
| domain_filled              |   count |
|:---------------------------|--------:|
| Product/Project Management |     193 |
| Engineering/IT             |     124 |
| Finance/Legal/HR           |      66 |

Top-3 regions (excluding Other/Not specified):
| region_display   |   count |
|:-----------------|--------:|
| Москва           |     124 |
| Russia           |      27 |
| Minsk, Belarus   |      26 |

Top-3 companies (excluding Other/Not specified):
| company_display        |   count |
|:-----------------------|--------:|
| Сбер                   |       4 |
| BostonGene             |       3 |
| Delta Distribution LLC |       3 |

### Key observations
- База содержит 730 профилей.
- Покрытие `cvEnhancedResult`: 72.2%.
- Покрытие `talentCard.jobs`: 72.2%.
- Покрытие `talentCard.overall_skills`: 72.2%.
- Покрытие `talentCard.specialist_category`: 71.4%.
- LaTeX-парсинг нашел `ExpHeader` у 71.6% пользователей.
- LaTeX skills section найдена у 72.2% пользователей.
- Гео-мэппинг схлопнул варианты Russia/Россия/РФ: 27 записей перешли в канон `Russia`.
- Industry анализируется только на subset с заполненным industry: 15.9% пользователей.
- `region=Not specified` остается у 33.0% базы; `company=Not specified` — у 27.8%.
- В fallback цепочку региона добавлено альтернативных geo-колонок: 0.
- Топ tools: Jira, Confluence, Sql, Miro, Figma.

## 2) Coverage / Parsing validation
| metric                               |   value |
|:-------------------------------------|--------:|
| users_total                          |   730   |
| coverage_cvEnhancedResult_%          |    72.2 |
| coverage_talentCard_jobs_%           |    72.2 |
| coverage_overall_skills_%            |    72.2 |
| coverage_specialist_category_%       |    71.4 |
| coverage_industry_talentCard_%       |    15.9 |
| users_with_latex_block               |   527   |
| share_users_with_latex_block_%       |    72.2 |
| users_with_expheader                 |   523   |
| share_users_with_expheader_%         |    71.6 |
| users_with_skills_section            |   527   |
| share_users_with_skills_section_%    |    72.2 |
| users_with_languages_section         |   421   |
| share_users_with_languages_section_% |    57.7 |
| users_with_education_section         |    39   |
| share_users_with_education_section_% |     5.3 |
| company_comparable_users             |   521   |
| current_company_matches              |   493   |
| current_company_match_rate_%         |    94.6 |

Columns inventory (real CSV structure + non-null profile + length stats):
- `outputs/tables/columns_inventory.csv`

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
<p><img src="outputs/figures/13_donut_seniority_filled.png" width="49%"><img src="outputs/figures/14_donut_cv_generation_language.png" width="49%"></p>

![Top-20 strata](outputs/figures/15_strata_top20.png)

Ключевые таблицы стратификации:
- `outputs/tables/strata_top20.csv`
- `outputs/tables/domain_distribution.csv`
- `outputs/tables/role_family_distribution.csv`
- `outputs/tables/seniority_distribution.csv`
- `outputs/tables/experience_bin_distribution.csv`
- `outputs/tables/leadership_distribution.csv`
- `outputs/tables/cv_generation_language_distribution.csv`

## 7) Not specified research
| field     |   total_missing_count |   share_missing_% |   share_filled_by_fallback | source_breakdown                                                                            |
|:----------|----------------------:|------------------:|---------------------------:|:--------------------------------------------------------------------------------------------|
| domain    |                     0 |               0   |                       28.6 | talentCard:71.4%; inferred:28.6%                                                            |
| region    |                   241 |              33   |                       20.7 | latex_expheader:46.3%; not_specified:33.0%; latex_header:14.0%; talentCard:6.7%             |
| company   |                   203 |              27.8 |                        0.5 | latex_expheader:71.6%; not_specified:27.8%; talentCard:0.5%                                 |
| seniority |                   293 |              40.1 |                       30.7 | not_specified:40.1%; talentCard:29.2%; inferred_job_title:16.6%; inferred_header_role:14.1% |
| industry  |                   614 |              84.1 |                        0   | not_specified:84.1%; talentCard:15.9%                                                       |

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

## 8) Domain Other
Исследование домена `Other` вынесено в отдельный отчёт: `REPORT_OTHERS.md`.

## 9) Appendix
Артефакты:
- Figures: `outputs/figures/*.png`
- Tables: `outputs/tables/*.csv`
- Other report: `REPORT_OTHERS.md` + `outputs/others/*`
- Notebook: `notebooks/mis_users_resume_bot.ipynb`
- Geo mapping audit: `outputs/tables/geo_mapping_audit.csv`
- Geo mapping top-50: `outputs/tables/geo_mapping_top50.csv`

How to reproduce:
```bash
python analytics/mis_users_resume_bot/src/build_mis.py \
  --input /mnt/data/prointerview-prod.users.csv \
  --base-dir analytics/mis_users_resume_bot
```
