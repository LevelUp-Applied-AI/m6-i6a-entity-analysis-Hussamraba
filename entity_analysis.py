"""
Module 6 Week A — Integration: Entity Analysis Pipeline
"""

import unicodedata
from collections import Counter
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt
import spacy
import seaborn as sns


def load_corpus(filepath="data/climate_articles.csv"):
    df = pd.read_csv(filepath)
    return df


def preprocess_corpus(df):
    df = df.copy()

    def normalize_text(text):
        if pd.isna(text):
            return ""
        return unicodedata.normalize("NFC", str(text))

    df["processed_text"] = df["text"].apply(normalize_text)

    # Arabic rows should not be processed with English NLP
    df.loc[df["language"] == "ar", "processed_text"] = ""

    return df


def run_ner_pipeline(df, nlp):
    rows = []

    english_df = df[df["language"] == "en"]

    for _, row in english_df.iterrows():
        doc = nlp(row["text"])

        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"]
    )


def aggregate_entity_stats(entity_df, articles_df):
    if entity_df.empty:
        return {
            "top_entities": pd.DataFrame(columns=["entity_text", "entity_label", "count"]),
            "label_counts": {},
            "co_occurrence": pd.DataFrame(columns=["entity_a", "entity_b", "co_count"]),
            "per_category": pd.DataFrame(columns=["category", "entity_label", "count"]),
        }

    top_entities = (
        entity_df
        .groupby(["entity_text", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )

    label_counts = entity_df["entity_label"].value_counts().to_dict()

    pair_counter = Counter()

    for _, group in entity_df.groupby("text_id"):
        unique_entities = sorted(set(group["entity_text"]))

        for entity_a, entity_b in combinations(unique_entities, 2):
            pair_counter[(entity_a, entity_b)] += 1

    co_occurrence = pd.DataFrame([
        {
            "entity_a": pair[0],
            "entity_b": pair[1],
            "co_count": count,
        }
        for pair, count in pair_counter.items()
        if count >= 2
    ])

    if not co_occurrence.empty:
        co_occurrence = (
            co_occurrence
            .sort_values("co_count", ascending=False)
            .head(50)
            .reset_index(drop=True)
        )
    else:
        co_occurrence = pd.DataFrame(columns=["entity_a", "entity_b", "co_count"])

    merged = entity_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id",
        right_on="id",
        how="left"
    )

    per_category = (
        merged
        .groupby(["category", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["category", "count"], ascending=[True, False])
    )

    return {
        "top_entities": top_entities,
        "label_counts": label_counts,
        "co_occurrence": co_occurrence,
        "per_category": per_category,
    }


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    top_entities = stats["top_entities"]

    if top_entities.empty:
        print("No entities to visualize.")
        return

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_entities,
        y="entity_text",
        x="count",
        hue="entity_label"
    )

    plt.title("Top 20 Named Entities in Climate Articles")
    plt.xlabel("Frequency")
    plt.ylabel("Entity")
    plt.legend(title="Entity Type")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_report(stats, co_occurrence):
    label_counts = stats["label_counts"]
    top_entities = stats["top_entities"].head(5)

    if co_occurrence is None:
        co_occurrence = pd.DataFrame(columns=["entity_a", "entity_b", "co_count"])

    top_pairs = co_occurrence.head(3)

    report = []

    report.append("Entity Analysis Report")
    report.append("=" * 50)

    report.append("\nEntity Counts by Type:")
    for label, count in label_counts.items():
        report.append(f"- {label}: {count}")

    report.append("\nTop 5 Most Frequent Entities:")
    for _, row in top_entities.iterrows():
        report.append(
            f"- {row['entity_text']} ({row['entity_label']}): {row['count']}"
        )

    report.append("\nTop 3 Co-occurring Entity Pairs:")
    if top_pairs.empty:
        report.append("- No repeated co-occurring pairs found.")
    else:
        for _, row in top_pairs.iterrows():
            report.append(
                f"- {row['entity_a']} + {row['entity_b']}: {row['co_count']}"
            )

    report.append("\nSummary:")
    report.append(
        "The entity analysis shows the most common organizations, locations, dates, "
        "and other named entities in the climate articles corpus. The label counts "
        "help identify which entity types dominate the dataset, while the co-occurrence "
        "pairs show which entities frequently appear together in the same articles. "
        "The per-category breakdown can also support comparisons between policy, "
        "science, impact, and adaptation articles."
    )

    return "\n".join(report)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    raw = load_corpus()

    if raw is not None:
        corpus = preprocess_corpus(raw)

        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            entities = run_ner_pipeline(corpus, nlp)

            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                stats = aggregate_entity_stats(entities, corpus)

                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print("\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print("\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    report = generate_report(stats, stats.get("co_occurrence"))

                    if report is not None:
                        print(f"\n{'=' * 50}")
                        print(report)