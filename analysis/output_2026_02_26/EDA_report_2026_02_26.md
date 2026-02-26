# EDA: Profiles Created on 2026-02-26 (UTC)

## Scope
- Source file: `snapshot_tue26.csv`
- Total profiles in source: **397**
- Profiles on target date `2026-02-26`: **170** (42.8% of source)

## Data Quality (Filtered Cohort)
```
tools                    94.7
years_experience         29.4
id                        0.0
updated_at_dt             0.0
employer_mentions         0.0
employer_type             0.0
region                    0.0
seniority                 0.0
role_family               0.0
full_name                 0.0
created_at_dt             0.0
first_name                0.0
specialist_category       0.0
cv_url                    0.0
updated_at                0.0
created_at                0.0
skills                    0.0
summary                   0.0
last_name                 0.0
has_explicit_employer     0.0
```

## Methodology (Inferred Fields)
- `role_family`: keyword-based mapping from `summary` first sentence + full text; fallback to `specialist_category`.
- `seniority`: title keyword rules (Executive/Lead/Senior/Middle/Junior) + years-of-experience fallback.
- `region`: geo keyword matching; if no explicit region and summary is Cyrillic, marked as `Russia & CIS (inferred)`.
- `employer_type`: keyword mapping by industry context in summary.
- `employer_mentions`: only explicit recognizable organization mentions from summary text.

## Key Metrics
- Profiles with parsed years-of-experience: **70.6%**
- Mean years-of-experience: **12.8**
- Median years-of-experience: **12.0**
- Profiles with explicit employer mentions: **7.1%**

### Role Family
```
                                            count  share_%
role_family                                               
Project / Program Management                   31     18.2
Operations / Administration / Supply Chain     19     11.2
Software Engineering                           19     11.2
Marketing / Sales / Growth                     18     10.6
Finance / Accounting / Audit                   17     10.0
DevOps / SRE / Infrastructure                  13      7.6
Product Management                             10      5.9
QA / Testing                                   10      5.9
Design / Creative                               8      4.7
Business / System Analysis                      6      3.5
Executive / General Management                  6      3.5
HR / People                                     5      2.9
Legal / Compliance                              4      2.4
Healthcare / Education                          3      1.8
Data / BI Analytics                             1      0.6
```

### Seniority
```
           count  share_%
seniority                
Lead          70     41.2
Senior        33     19.4
Executive     28     16.5
Unknown       16      9.4
Middle        16      9.4
Junior         7      4.1
```

### Region
```
                         count  share_%
region                                 
Russia & CIS (inferred)     75     44.1
Not specified               74     43.5
Global / Multi-region        8      4.7
Middle East                  5      2.9
Russia & CIS                 4      2.4
Europe                       4      2.4
```

### Employer Type
```
                                  count  share_%
employer_type                                   
IT Product / SaaS                    44     25.9
Not specified                        36     21.2
Banking / Fintech                    23     13.5
Telecom                              18     10.6
System Integration / Outsourcing     10      5.9
Retail / E-commerce / FMCG            9      5.3
Industrial / Manufacturing            9      5.3
Government / Public Sector            6      3.5
Education / EdTech                    5      2.9
Logistics / Transport                 3      1.8
Construction / Real Estate            3      1.8
Healthcare / Pharma                   3      1.8
Media / Gaming                        1      0.6
```

### Top Explicit Employer Mentions
```
employer_mentions
IBM                                3
Rostelecom                         2
Ministry of Digital Development    2
Minpromtorg                        2
VTB                                2
EMIAS                              1
Forward                            1
RZD                                1
EPAM                               1
```

### Top Skills (Keyword-level)
```
Jira                         39
Confluence                   35
Miro                         20
SQL                          19
Kanban                       18
Stakeholder Management       15
Power BI                     14
Python                       13
Agile (Scrum                 13
MS Project                   13
Waterfall                    13
Figma                        12
Grafana                      11
Git                          11
CI/CD                        11
PostgreSQL                   11
Docker                       10
Agile/Scrum                   9
Управление изменениями        9
Stakeholder management        9
TypeScript                    8
Операционное управление       8
Notion                        8
Trello                        8
Управление проектами          7
RabbitMQ                      7
Kubernetes                    7
CRM                           7
Управление стейкхолдерами     7
JavaScript                    7
```

## Visualizations
1. `figures/01_created_date_context.png`
2. `figures/02_specialist_category.png`
3. `figures/03_role_family.png`
4. `figures/04_seniority.png`
5. `figures/05_region.png`
6. `figures/06_employer_type.png`
7. `figures/07_employer_mentions.png` (if explicit mentions exist)
8. `figures/08_experience_distribution.png`
9. `figures/09_role_seniority_heatmap.png`
10. `figures/10_top_skills.png`
