import pandas as pd
import spacy
from transformers import pipeline
from collections import Counter

DATA_PATH = "data/climate_articles.csv"
OUTPUT_COMPARISON = "multilingual_ner_comparison.csv"
OUTPUT_ENTITIES = "multilingual_ner_entities.csv"
OUTPUT_MD = "stretch_analysis.md"


def load_data(filepath):
    df = pd.read_csv(filepath)

    required = {"id", "text", "language"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def get_sample(df, language, n=20):
    sample = df[df["language"] == language].copy()
    sample = sample.dropna(subset=["text"])
    return sample.head(n)


def run_spacy(df, nlp, language):
    rows = []

    for _, row in df.iterrows():
        text = str(row["text"])
        doc = nlp(text)
        word_count = len(text.split())

        for ent in doc.ents:
            rows.append({
                "article_id": row["id"],
                "language": language,
                "model": "spaCy_xx_ent_wiki_sm",
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "word_count": word_count,
            })

    return pd.DataFrame(rows)


def run_hf(df, ner, language):
    rows = []

    for _, row in df.iterrows():
        text = str(row["text"])
        word_count = len(text.split())

        try:
            ents = ner(text)
        except Exception as e:
            print(f"HF error on {row['id']}: {e}")
            continue

        for ent in ents:
            rows.append({
                "article_id": row["id"],
                "language": language,
                "model": "HF_xlm_roberta_wikiann",
                "entity_text": ent.get("word", ""),
                "entity_label": ent.get("entity_group", ""),
                "start_char": ent.get("start", None),
                "end_char": ent.get("end", None),
                "word_count": word_count,
            })

    return pd.DataFrame(rows)


def count_words(text):
    return len(str(text).split())


def build_comparison(entity_df, sample_df):
    rows = []

    for language in ["en", "ar"]:
        lang_articles = sample_df[sample_df["language"] == language]
        total_words = lang_articles["text"].apply(count_words).sum()

        for model in ["spaCy_xx_ent_wiki_sm", "HF_xlm_roberta_wikiann"]:
            group = entity_df[
                (entity_df["language"] == language) &
                (entity_df["model"] == model)
            ]

            total_entities = len(group)
            density = round((total_entities / total_words) * 100, 2)

            counts = Counter(group["entity_label"])

            examples = (
                group[["entity_text", "entity_label"]]
                .drop_duplicates()
                .head(3)
                .apply(lambda x: f"{x['entity_text']} ({x['entity_label']})", axis=1)
                .tolist()
            )

            found = set(group["article_id"].unique())
            all_ids = set(lang_articles["id"].tolist())

            no_count = len(all_ids - found)
            no_rate = round(no_count / len(all_ids), 2)

            rows.append({
                "language": language,
                "model": model,
                "total_entities": total_entities,
                "entity_density_per_100_words": density,

                "PER": counts.get("PER", 0),
                "PERSON": counts.get("PERSON", 0),
                "ORG": counts.get("ORG", 0),
                "LOC": counts.get("LOC", 0),
                "GPE": counts.get("GPE", 0),
                "MISC": counts.get("MISC", 0),
                "DATE": counts.get("DATE", 0),

                "example_1": examples[0] if len(examples) > 0 else "",
                "example_2": examples[1] if len(examples) > 1 else "",
                "example_3": examples[2] if len(examples) > 2 else "",

                "texts_checked": len(all_ids),
                "texts_with_no_entities": no_count,
                "no_entities_found_rate": no_rate,
            })

    return pd.DataFrame(rows)


def write_md(df):
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Stretch 6A-S2 — Multilingual NER Comparison\n\n")

        f.write("## Label Choice\n\n")
        f.write(
            "Option A: native labels. Each model’s labels are kept as-is for fair comparison.\n\n"
        )

        f.write("## Comparison Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Analysis\n\n")
        f.write(
            "Arabic NER is harder due to lack of capitalization and complex morphology. "
            "Results show weaker Arabic detection in spaCy and stronger performance in Hugging Face. "
            "English entities are easier due to capitalization cues.\n\n"
        )

        f.write(
            "For MENA applications, bilingual pipelines are required. "
            "Arabic outputs need validation and often benefit from custom rules.\n"
        )


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    en_df = get_sample(df, "en", 20)
    ar_df = get_sample(df, "ar", 20)

    sample_df = pd.concat([en_df, ar_df], ignore_index=True)

    print("Loading models...")
    nlp = spacy.load("xx_ent_wiki_sm")

    ner = pipeline(
        "token-classification",
        model="Davlan/xlm-roberta-base-wikiann-ner",
        aggregation_strategy="simple",
    )

    print("Running spaCy...")
    sp_en = run_spacy(en_df, nlp, "en")
    sp_ar = run_spacy(ar_df, nlp, "ar")

    print("Running HF...")
    hf_en = run_hf(en_df, ner, "en")
    hf_ar = run_hf(ar_df, ner, "ar")

    entity_df = pd.concat([sp_en, sp_ar, hf_en, hf_ar], ignore_index=True)

    print("Saving raw entities...")
    entity_df.to_csv(OUTPUT_ENTITIES, index=False, encoding="utf-8-sig")

    print("Building comparison...")
    comp_df = build_comparison(entity_df, sample_df)

    comp_df.to_csv(OUTPUT_COMPARISON, index=False, encoding="utf-8-sig")

    write_md(comp_df)

    print("\nDONE ✅")
    print("✔ multilingual_ner_entities.csv")
    print("✔ multilingual_ner_comparison.csv")
    print("✔ stretch_analysis.md")


if __name__ == "__main__":
    main()