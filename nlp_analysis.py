import os
import re
from collections import Counter

import nltk
import pandas as pd
import spacy
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline


def download_nltk_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "vader_lexicon",
    ]

    for resource in resources:
        nltk.download(resource, quiet=True)


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def tokenize_and_preprocess(text, stop_words):
    tokens = word_tokenize(clean_text(text))

    return [
        token
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]


def nltk_named_entities(text):
    entities = []

    for sentence in sent_tokenize(text):
        tagged_words = pos_tag(word_tokenize(sentence))
        entity_tree = ne_chunk(tagged_words)

        for subtree in entity_tree:
            if hasattr(subtree, "label"):
                entity_text = " ".join(
                    word for word, tag in subtree.leaves()
                )
                entities.append((entity_text, subtree.label()))

    return entities


def vader_sentiment(text, analyzer):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, score


def transformer_sentiment(text, classifier):
    result = classifier(text[:512])[0]
    return result["label"].lower(), round(result["score"], 4)


def main():
    download_nltk_resources()

    dataset_path = "SBA927.txt"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} text record(s).")

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    vader_analyzer = SentimentIntensityAnalyzer()

    print("Loading spaCy English model...")
    spacy_model = spacy.load("en_core_web_sm")

    print("Loading pre-trained sentiment model...")
    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    analysis_rows = []
    all_lemmatized_tokens = []
    all_entities = []

    for index, text in enumerate(dataset, start=1):
        processed_tokens = tokenize_and_preprocess(text, stop_words)
        stemmed_tokens = [stemmer.stem(token) for token in processed_tokens]
        lemmatized_tokens = [
            lemmatizer.lemmatize(token)
            for token in processed_tokens
        ]

        nltk_pos_result = pos_tag(word_tokenize(text))
        nltk_ner_result = nltk_named_entities(text)

        doc = spacy_model(text)

        spacy_pos_result = [
            (token.text, token.pos_)
            for token in doc
            if not token.is_space
        ]

        spacy_entities = [
            (entity.text, entity.label_)
            for entity in doc.ents
        ]

        vader_label, vader_score = vader_sentiment(text, vader_analyzer)
        transformer_label, transformer_score = transformer_sentiment(
            text,
            sentiment_classifier,
        )

        all_lemmatized_tokens.extend(lemmatized_tokens)
        all_entities.extend(spacy_entities)

        row = {
            "text_number": index,
            "original_text": text,
            "cleaned_text": clean_text(text),
            "processed_tokens": ", ".join(processed_tokens),
            "stemmed_tokens": ", ".join(stemmed_tokens),
            "lemmatized_tokens": ", ".join(lemmatized_tokens),
            "nltk_pos_tags": str(nltk_pos_result),
            "nltk_named_entities": str(nltk_ner_result),
            "spacy_pos_tags": str(spacy_pos_result),
            "spacy_named_entities": str(spacy_entities),
            "vader_sentiment": vader_label,
            "vader_compound_score": vader_score,
            "transformer_sentiment": transformer_label,
            "transformer_confidence": transformer_score,
        }

        analysis_rows.append(row)

        print(f"\n--- Text {index} ---")
        print("Original:", text)
        print("Processed tokens:", processed_tokens)
        print("NLTK named entities:", nltk_ner_result)
        print("spaCy named entities:", spacy_entities)
        print("VADER sentiment:", vader_label, vader_score)
        print("Transformer sentiment:", transformer_label, transformer_score)

    results_df = pd.DataFrame(analysis_rows)

    results_path = os.path.join(output_folder, "nlp_results.csv")
    results_df.to_csv(results_path, index=False)

    top_terms = Counter(all_lemmatized_tokens).most_common(20)
    top_entities = Counter(all_entities).most_common(20)

    summary_path = os.path.join(output_folder, "business_insights.txt")

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("SBA 927 Business Text Analytics Summary\n")
        file.write("=" * 45 + "\n\n")
        file.write(f"Number of text records: {len(dataset)}\n\n")

        file.write("Top 20 Lemmatized Terms:\n")
        for term, count in top_terms:
            file.write(f"- {term}: {count}\n")

        file.write("\nTop Named Entities from spaCy:\n")
        for entity, count in top_entities:
            file.write(f"- {entity}: {count}\n")

        file.write("\nVADER Sentiment Counts:\n")
        file.write(results_df["vader_sentiment"].value_counts().to_string())

        file.write("\n\nTransformer Sentiment Counts:\n")
        file.write(
            results_df["transformer_sentiment"]
            .value_counts()
            .to_string()
        )

    print("\nAnalysis complete.")
    print(f"CSV results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()