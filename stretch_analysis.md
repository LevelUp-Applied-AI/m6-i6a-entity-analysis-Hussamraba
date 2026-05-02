# Stretch 6A-S2 — Multilingual NER Comparison

## Label Choice

Option A: native labels. Each model’s labels are kept as-is for fair comparison.

## Comparison Table

| language   | model                  |   total_entities |   entity_density_per_100_words |   PER |   PERSON |   ORG |   LOC |   GPE |   MISC |   DATE | example_1                                          | example_2                      | example_3          |   texts_checked |   texts_with_no_entities |   no_entities_found_rate |
|:-----------|:-----------------------|-----------------:|-------------------------------:|------:|---------:|------:|------:|------:|-------:|-------:|:---------------------------------------------------|:-------------------------------|:-------------------|----------------:|-------------------------:|-------------------------:|
| en         | spaCy_xx_ent_wiki_sm   |               93 |                           7.62 |    12 |        0 |    39 |    27 |     0 |     15 |      0 | IPCC (MISC)                                        | Sixth Assessment Report (MISC) | Celsius (PER)      |              20 |                        0 |                     0    |
| en         | HF_xlm_roberta_wikiann |               93 |                           7.62 |    10 |        0 |    55 |    28 |     0 |      0 |      0 | Antonio Guterres (PER)                             | COP (ORG)                      | Dubai (LOC)        |              20 |                        0 |                     0    |
| ar         | spaCy_xx_ent_wiki_sm   |               20 |                           2.15 |     9 |        0 |     2 |     2 |     0 |      7 |      0 | وأكد التقرير (PER)                                 | وقّع الأردن (MISC)             | وأكد وزير (PER)    |              20 |                        5 |                     0.25 |
| ar         | HF_xlm_roberta_wikiann |               64 |                           6.88 |     2 |        0 |    33 |    29 |     0 |      0 |      0 | الهيئة الحكومية الدولية المعنية بتغير المناخ (ORG) | الأردن (LOC)                   | البنك الدولي (ORG) |              20 |                        0 |                     0    |

## Analysis

Arabic NER is harder due to lack of capitalization and complex morphology. Results show weaker Arabic detection in spaCy and stronger performance in Hugging Face. English entities are easier due to capitalization cues.

For MENA applications, bilingual pipelines are required. Arabic outputs need validation and often benefit from custom rules.
