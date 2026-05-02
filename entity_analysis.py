import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter


def load_corpus(filepath="data/climate_articles.csv"):
    return pd.read_csv(filepath)


def preprocess_corpus(df):
    df = df.copy()

    def normalize(text):
        if isinstance(text, str):
            return unicodedata.normalize("NFC", text)
        return ""

    df["processed_text"] = df["text"].apply(normalize)
    return df


def run_ner_pipeline(df, nlp):
    rows = []

    en_df = df[df["language"] == "en"]

    for _, row in en_df.iterrows():
        doc = nlp(row["text"])

        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(rows)


def aggregate_entity_stats(entity_df, articles_df):
    # Top entities
    top_entities = (
        entity_df.groupby(["entity_text", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .head(20)
    )

    # Label counts
    label_counts = dict(Counter(entity_df["entity_label"]))

    # Co-occurrence
    pairs = []
    for text_id in entity_df["text_id"].unique():
        ents = entity_df[entity_df["text_id"] == text_id]["entity_text"].unique()
        pairs.extend(list(itertools.combinations(ents, 2)))

    if pairs:
        co_occurrence = (
            pd.DataFrame(pairs, columns=["entity_a", "entity_b"])
            .value_counts()
            .reset_index(name="co_count")
        )
    else:
        co_occurrence = pd.DataFrame(columns=["entity_a", "entity_b", "co_count"])

    # Per category
    merged = entity_df.merge(articles_df, left_on="text_id", right_on="id")

    per_category = (
        merged.groupby(["category", "entity_label"])
        .size()
        .reset_index(name="count")
    )

    return {
        "top_entities": top_entities,
        "label_counts": label_counts,
        "co_occurrence": co_occurrence,
        "per_category": per_category,
    }


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    df = stats["top_entities"].head(10)

    plt.figure()
    plt.bar(df["entity_text"], df["count"])
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(stats, co_occurrence):
    report = "Entity Analysis Report\n\n"

    report += "Top Entities:\n"
    report += str(stats["top_entities"].head(5)) + "\n\n"

    report += "Label Counts:\n"
    report += str(stats["label_counts"]) + "\n\n"

    report += "Co-occurrence:\n"
    report += str(co_occurrence.head(3)) + "\n\n"

    report += "Summary:\nEntity extraction completed successfully."

    return report