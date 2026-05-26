import argparse
import sys
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline


def load_models():
    vader = SentimentIntensityAnalyzer()
    transformer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    return vader, transformer


def analyze_vader(text, vader):
    scores = vader.polarity_scores(text)
    return scores["compound"]


def analyze_textblob(text):
    return TextBlob(text).sentiment.polarity


def analyze_transformer(text, transformer):
    result = transformer(text[:512])[0]
    score = result["score"]

    # Convert confidence into probability of POSITIVE class
    if result["label"] == "POSITIVE":
        p_positive = score
    else:
        p_positive = 1 - score

    # Map [0, 1] probability to [-1, 1] sentiment scale
    sentiment = 2 * p_positive - 1
    return round(sentiment, 4)


def analyze_text(text, vader, transformer):
    return {
        "VADER": analyze_vader(text, vader),
        "TextBlob": analyze_textblob(text),
        "Transformer": analyze_transformer(text, transformer),
    }


def process_file(input_path, output_path, vader, transformer):
    if input_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        print(f"Unsupported file format: {input_path}")
        sys.exit(1)

    tweet_col = None
    for col in df.columns:
        if col.strip().lower() == "tweet":
            tweet_col = col
            break

    if tweet_col is None:
        print(f"No 'Tweet' column found. Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {input_path}")
    print("Running sentiment analysis...\n")

    vader_scores = []
    textblob_scores = []
    transformer_scores = []

    for i, text in enumerate(df[tweet_col]):
        text = str(text)
        vader_scores.append(analyze_vader(text, vader))
        textblob_scores.append(analyze_textblob(text))
        transformer_scores.append(analyze_transformer(text, transformer))

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            print(f"  Processed {i + 1}/{len(df)} tweets")

    df["VADER_Score"] = vader_scores
    df["TextBlob_Score"] = textblob_scores
    df["Transformer_Score"] = transformer_scores

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


def process_text(text, vader, transformer):
    scores = analyze_text(text, vader, transformer)
    print(f"\nText: {text}\n")
    print(f"{'Method':<15} {'Score':>10}  {'Sentiment':<10}")
    print("-" * 40)
    for method, score in scores.items():
        if score > 0.05:
            label = "Positive"
        elif score < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        print(f"{method:<15} {score:>10.4f}  {label:<10}")


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analyzer")
    parser.add_argument(
        "input",
        help="Text string to analyze, or path to a CSV/Excel file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV path (default: <input>_scored.csv)",
        default=None,
    )
    args = parser.parse_args()

    print("Loading models...")
    vader, transformer = load_models()
    print("Models loaded.\n")

    if args.input.endswith((".csv", ".xlsx", ".xls")):
        output = args.output or args.input.rsplit(".", 1)[0] + "_scored.csv"
        process_file(args.input, output, vader, transformer)
    else:
        process_text(args.input, vader, transformer)


if __name__ == "__main__":
    main()
