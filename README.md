## Integration 6A — Entity Analysis Pipeline

Completed the full entity analysis pipeline for the climate articles corpus.

### What I Completed

- Loaded the climate articles dataset.
- Added language-aware preprocessing using Unicode NFC normalization.
- Preserved the raw `text` column for NER.
- Filtered the corpus to English articles only before running spaCy NER.
- Extracted named entities with `text_id`, `entity_text`, `entity_label`, `start_char`, and `end_char`.
- Aggregated:
  - Top 20 entities
  - Entity label counts
  - Co-occurring entity pairs
  - Per-category entity distribution
- Created a visualization file: `entity_distribution.png`.
- Generated a structured text report from the entity statistics.

### Results Summary

The corpus contains 200 articles:
- English: 132
- Arabic: 68

The pipeline extracted **1202 entities** from the English articles.

Top entity types:
- DATE: 256
- ORG: 184
- GPE: 165
- CARDINAL: 138
- PERCENT: 103

Top entities:
- 2030 (DATE): 25
- 2023 (DATE): 21
- Jordan (GPE): 16
- annually (DATE): 15
- annual (DATE): 10

### Key Findings

The corpus focuses heavily on dates, organizations, and geographic locations. This makes sense for climate articles because they often discuss climate targets, reporting years, countries, regions, and institutions.

### Category Patterns

Adaptation articles had high counts for DATE, GPE, ORG, CARDINAL, and PERCENT entities. This suggests that adaptation texts often focus on timelines, locations, organizations, and measurable climate impacts.

### Co-occurrence Insights

The strongest co-occurring entity pairs were:

- Jordan + annually: 7
- 2030 + Jordan: 5
- 2023 + 2030: 5

These patterns suggest that Jordan is often discussed together with annual reporting and future climate goals.

### Dashboard Recommendation

I recommend building an entity relationship dashboard for climate researchers. The dashboard would show the most frequent organizations, locations, dates, and their co-occurrence patterns across article categories. This would help researchers quickly understand which countries, institutions, and climate targets are most connected in the corpus.