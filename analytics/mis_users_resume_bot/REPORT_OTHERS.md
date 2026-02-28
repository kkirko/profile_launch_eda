# REPORT_OTHERS: Domain Other Deep Dive

## 1) Size
- Other users: **198**
- Share of all users: **27.4%**
- Other users with `cvEnhancedResult`: **1**
- Share among users with `cvEnhancedResult`: **0.2%**

## 2) Coverage
| metric                               |   value |
|:-------------------------------------|--------:|
| other_users_total                    |   198   |
| share_all_users_%                    |    27.4 |
| other_users_with_cvEnhancedResult    |     1   |
| share_among_cvEnhancedResult_users_% |     0.2 |
| jobs_latex_present_%                 |     0.5 |
| jobs_talent_present_%                |     0.5 |
| skills_or_tools_present_%            |     0.5 |
| region_specified_%                   |     0.5 |
| seniority_specified_%                |     0.5 |

## 3) Moved out of Other by remapping
| old_label_keywords                                      | new_domain                 |   count |
|:--------------------------------------------------------|:---------------------------|--------:|
| Other/Not specified -> Engineering keywords             | Engineering/IT             |       4 |
| Other/Not specified -> Business/System Analyst keywords | Product/Project Management |       2 |
| Other/Not specified -> Recruiter/TA/HR keywords         | Finance/Legal/HR           |       2 |
| Other/Not specified -> generic remapping                | Finance/Legal/HR           |       2 |
| Other/Not specified -> generic remapping                | Product/Project Management |       2 |
| Other/Not specified -> Data/Analytics keywords          | Product/Project Management |       1 |
| Other/Not specified -> UX/UI/Design keywords            | Design/Creative            |       1 |
| Other/Not specified -> generic remapping                | Marketing/Sales            |       1 |

## 4) Who Are These Candidates
Top-30 titles across all jobs history:
| job_title                       |   count |   share_% |
|:--------------------------------|--------:|----------:|
| Director of Corporate Property  |       2 |      33.3 |
| Managing Director               |       2 |      33.3 |
| Head of Real Estate Development |       2 |      33.3 |

Top-20 current titles:
| current_job_title_filled        |   count |   share_% |
|:--------------------------------|--------:|----------:|
| Not specified                   |     197 |      99.5 |
| Head of Real Estate Development |       1 |       0.5 |

Top-30 companies across all jobs history:
| company                    |   count |   share_% |
|:---------------------------|--------:|----------:|
| GAZ Gorky Automotive Plant |       2 |      33.3 |
| Abris Development Group    |       2 |      33.3 |
| Abris Management           |       2 |      33.3 |

Most frequent title keywords:
| title_keyword                   |   count |   share_% |
|:--------------------------------|--------:|----------:|
| director of corporate property  |       2 |      33.3 |
| managing director               |       2 |      33.3 |
| head of real estate development |       2 |      33.3 |

## 5) Skills & Stack
Top tools:
| tool               |   count |   share_% |
|:-------------------|--------:|----------:|
| BIM & Digital Twin |       1 |       100 |

Top skills:
| skill                                    |   count |   share_% |
|:-----------------------------------------|--------:|----------:|
| Owner-Side Development Management        |       1 |         5 |
| Feasibility & Due Diligence              |       1 |         5 |
| Envision                                 |       1 |         5 |
| WELL                                     |       1 |         5 |
| LEED                                     |       1 |         5 |
| ITACA                                    |       1 |         5 |
| Green Building (CasaClima                |       1 |         5 |
| Value Engineering                        |       1 |         5 |
| Building Performance & Energy Efficiency |       1 |         5 |
| Asset & Property Management              |       1 |         5 |
| Commissioning & Handover                 |       1 |         5 |
| Construction Supervision                 |       1 |         5 |
| Schedule & Risk Control                  |       1 |         5 |
| Cost                                     |       1 |         5 |
| Contract Administration                  |       1 |         5 |
| Procurement & Tendering                  |       1 |         5 |
| Permitting & Entitlements                |       1 |         5 |
| Design Management                        |       1 |         5 |
| Concept & Business Planning              |       1 |         5 |
| Stakeholder Management                   |       1 |         5 |

## 6) Geography & Seniority
Top regions:
| region_norm   |   count |   share_% |
|:--------------|--------:|----------:|
| Not specified |     197 |      99.5 |
| Rome, Italy   |       1 |       0.5 |

Seniority:
| seniority_filled   |   count |   share_% |
|:-------------------|--------:|----------:|
| Not specified      |     197 |      99.5 |
| C-level            |       1 |       0.5 |

## 7) Rule-Based Buckets
| bucket        |   count |   share_% |
|:--------------|--------:|----------:|
| Other-unknown |     197 |      99.5 |
| SWE/IT        |       1 |       0.5 |

## 8) Figures
![Top titles across all jobs](outputs/others/figures/01_other_titles_history_top30.png)
![Top companies across all jobs](outputs/others/figures/02_other_companies_history_top30.png)
![Rule-based buckets](outputs/others/figures/04_other_bucket_distribution.png)

## 9) Appendix
- Tables: `outputs/others/tables/*.csv`
- Figures: `outputs/others/figures/*.png`
