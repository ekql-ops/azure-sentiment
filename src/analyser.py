"""
╔══════════════════════════════════════════════════════════════╗
║         Azure Sentiment Analyser — Text Analytics Tool       ║
║                    Abubakar Albakri, 2026                    ║
╚══════════════════════════════════════════════════════════════╝

Uses Microsoft Azure Cognitive Services (Text Analytics API) to
perform sentiment analysis, key phrase extraction, and language
detection on text input — via CLI or batch file processing.

Demonstrates real Azure cloud integration (AZ-900 / AI-900).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
except ImportError:
    print("[ERROR] Missing Azure SDK. Run:")
    print("  pip install azure-ai-textanalytics")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


# ─── Colour helpers (cross-platform) ──────────────────────────
def _c(code: str, text: str) -> str:
    """Wrap text in ANSI colour code."""
    if os.name == "nt":
        return text  # Skip colours on Windows Command Prompt
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c("32", t)
RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ─── Azure Client ─────────────────────────────────────────────

def get_client() -> TextAnalyticsClient:
    """
    Build and return an authenticated Azure Text Analytics client.

    Credentials are loaded from environment variables:
        AZURE_LANGUAGE_ENDPOINT  — your Azure endpoint URL
        AZURE_LANGUAGE_KEY       — your Azure API key

    Set these in a .env file or export them in your shell.
    """
    endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT", "").rstrip("/")
    key      = os.getenv("AZURE_LANGUAGE_KEY", "")

    if not endpoint or not key:
        print(RED("\n  [✗] Azure credentials not set."))
        print("  Set the following environment variables:")
        print("    AZURE_LANGUAGE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com")
        print("    AZURE_LANGUAGE_KEY=<your-api-key>\n")
        print("  Or create a .env file with these values.\n")
        sys.exit(1)

    return TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )


# ─── Analysis Functions ───────────────────────────────────────

def analyse_sentiment(client: TextAnalyticsClient, texts: list[str]) -> list[dict]:
    """
    Run sentiment analysis on a list of texts.
    Returns structured results with overall sentiment and confidence scores.
    """
    results = []
    try:
        response = client.analyze_sentiment(
            documents=texts,
            show_opinion_mining=True
        )
        for i, doc in enumerate(response):
            if doc.is_error:
                results.append({"error": doc.error.message, "text": texts[i]})
                continue

            # Extract sentence-level sentiments
            sentences = []
            for sentence in doc.sentences:
                sentence_data = {
                    "text":      sentence.text,
                    "sentiment": sentence.sentiment,
                    "scores": {
                        "positive": round(sentence.confidence_scores.positive, 4),
                        "neutral":  round(sentence.confidence_scores.neutral,  4),
                        "negative": round(sentence.confidence_scores.negative, 4),
                    }
                }
                # Opinion mining (aspect-based sentiment)
                opinions = []
                for mined_opinion in sentence.mined_opinions:
                    target = mined_opinion.target
                    assessments = [
                        {
                            "text":      a.text,
                            "sentiment": a.sentiment,
                            "positive":  round(a.confidence_scores.positive, 4),
                            "negative":  round(a.confidence_scores.negative, 4),
                        }
                        for a in mined_opinion.assessments
                    ]
                    opinions.append({"target": target.text, "assessments": assessments})
                sentence_data["opinions"] = opinions
                sentences.append(sentence_data)

            results.append({
                "text":      texts[i],
                "sentiment": doc.sentiment,
                "scores": {
                    "positive": round(doc.confidence_scores.positive, 4),
                    "neutral":  round(doc.confidence_scores.neutral,  4),
                    "negative": round(doc.confidence_scores.negative, 4),
                },
                "sentences": sentences,
            })

    except HttpResponseError as e:
        print(RED(f"  [✗] Azure API error: {e.message}"))
        sys.exit(1)

    return results


def extract_key_phrases(client: TextAnalyticsClient, texts: list[str]) -> list[list[str]]:
    """Extract key phrases from a list of texts."""
    try:
        response = client.extract_key_phrases(documents=texts)
        return [
            doc.key_phrases if not doc.is_error else []
            for doc in response
        ]
    except HttpResponseError as e:
        print(RED(f"  [✗] Key phrase extraction error: {e.message}"))
        return [[] for _ in texts]


def detect_language(client: TextAnalyticsClient, texts: list[str]) -> list[dict]:
    """Detect the language of each text."""
    try:
        response = client.detect_language(documents=texts)
        return [
            {
                "name": doc.primary_language.name,
                "iso":  doc.primary_language.iso6391_name,
                "confidence": round(doc.primary_language.confidence_score, 4),
            }
            if not doc.is_error else {"name": "Unknown", "iso": "?", "confidence": 0}
            for doc in response
        ]
    except HttpResponseError as e:
        print(RED(f"  [✗] Language detection error: {e.message}"))
        return [{"name": "Unknown", "iso": "?", "confidence": 0} for _ in texts]


# ─── Display ─────────────────────────────────────────────────

def sentiment_bar(positive: float, neutral: float, negative: float, width: int = 30) -> str:
    """Render a colour-coded sentiment bar."""
    pos_w = int(positive * width)
    neu_w = int(neutral  * width)
    neg_w = width - pos_w - neu_w

    bar  = GREEN("█" * pos_w)
    bar += YELLOW("█" * neu_w)
    bar += RED("█" * max(0, neg_w))
    return f"[{bar}]"


def sentiment_icon(sentiment: str) -> str:
    icons = {"positive": "😊", "neutral": "😐", "negative": "😞", "mixed": "🤔"}
    return icons.get(sentiment.lower(), "❓")


def print_result(result: dict, key_phrases: list[str], language: dict, index: int):
    """Pretty-print a single analysis result."""
    if "error" in result:
        print(RED(f"\n  [{index}] Error: {result['error']}"))
        return

    sentiment = result["sentiment"]
    scores    = result["scores"]
    icon      = sentiment_icon(sentiment)

    print(f"\n  ┌─ Result {index} {'─' * 50}")
    print(f"  │  Text       : {DIM(result['text'][:80])}")
    print(f"  │  Language   : {language['name']} ({language['iso']}) — {language['confidence']*100:.0f}% confidence")
    print(f"  │  Sentiment  : {icon}  {BOLD(sentiment.upper())}")
    print(f"  │  Bar        : {sentiment_bar(scores['positive'], scores['neutral'], scores['negative'])}")
    print(f"  │             :  {GREEN('+')} {scores['positive']*100:5.1f}%  {YELLOW('~')} {scores['neutral']*100:5.1f}%  {RED('-')} {scores['negative']*100:5.1f}%")

    if key_phrases:
        print(f"  │  Key Phrases: {', '.join(key_phrases[:6])}")

    # Sentence breakdown
    if len(result["sentences"]) > 1:
        print(f"  │")
        print(f"  │  Sentence breakdown:")
        for s in result["sentences"]:
            s_icon = sentiment_icon(s["sentiment"])
            preview = s["text"][:55] + "..." if len(s["text"]) > 55 else s["text"]
            print(f"  │    {s_icon}  {preview}")
            # Show opinions if found
            for op in s.get("opinions", []):
                assessments = ", ".join(
                    f"{a['text']} ({a['sentiment']})"
                    for a in op["assessments"]
                )
                print(f"  │       ↳ {op['target']}: {assessments}")

    print(f"  └{'─' * 57}")


def print_batch_summary(results: list[dict]):
    """Print aggregate statistics for a batch analysis."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    counts = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
    for r in valid:
        counts[r["sentiment"]] = counts.get(r["sentiment"], 0) + 1

    total = len(valid)
    print(f"\n  ── Batch Summary ({total} texts) ──────────────────────────")
    for sentiment, count in counts.items():
        pct  = count / total * 100
        bar  = "█" * int(pct / 5)
        icon = sentiment_icon(sentiment)
        print(f"  {icon}  {sentiment:<10}  {count:>3}  {bar:<20}  {pct:.1f}%")
    print()


# ─── Batch File Processing ────────────────────────────────────

def process_file(path: str, client: TextAnalyticsClient, save_json: bool = False):
    """
    Read texts from a file (one per line) and analyse them in batches.
    Azure Text Analytics supports up to 10 documents per request.
    """
    p = Path(path)
    if not p.exists():
        print(RED(f"  [✗] File not found: {path}"))
        return

    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print(YELLOW("  [!] File is empty."))
        return

    print(f"\n  Processing {len(lines)} texts from '{p.name}'...")

    all_results = []
    batch_size  = 10

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        print(DIM(f"  Batch {i // batch_size + 1}..."))

        sentiments  = analyse_sentiment(client, batch)
        key_phrases = extract_key_phrases(client, batch)
        languages   = detect_language(client, batch)

        for j, (result, kp, lang) in enumerate(zip(sentiments, key_phrases, languages)):
            print_result(result, kp, lang, i + j + 1)
            all_results.append({**result, "key_phrases": kp, "language": lang})

        if i + batch_size < len(lines):
            time.sleep(0.5)  # Avoid rate limiting

    print_batch_summary(all_results)

    if save_json:
        out = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(out).write_text(json.dumps(all_results, indent=2))
        print(f"  [✓] Results saved to: {out}\n")


# ─── Interactive Mode ─────────────────────────────────────────

def interactive_mode(client: TextAnalyticsClient):
    """Run analysis on text entered directly by the user."""
    print(BOLD("\n  ── Interactive Mode ──"))
    print("  Enter text to analyse (or 'quit' to exit):\n")

    count = 1
    while True:
        try:
            text = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.\n")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("  Goodbye.\n")
            break

        if not text:
            continue

        sentiments  = analyse_sentiment(client, [text])
        key_phrases = extract_key_phrases(client, [text])
        languages   = detect_language(client, [text])

        print_result(sentiments[0], key_phrases[0], languages[0], count)
        count += 1


# ─── CLI Entrypoint ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Azure Sentiment Analyser — Text Analytics CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyser.py                          # Interactive mode
  python analyser.py -t "I love Python!"     # Analyse a single text
  python analyser.py -f texts.txt            # Analyse from file
  python analyser.py -f texts.txt --save     # Analyse + save JSON results
        """
    )
    parser.add_argument("-t", "--text", help="Single text to analyse")
    parser.add_argument("-f", "--file", help="Path to a text file (one entry per line)")
    parser.add_argument("--save",   action="store_true", help="Save results as JSON")
    args = parser.parse_args()

    print(BOLD("\n  ╔══════════════════════════════════════╗"))
    print(BOLD("  ║   Azure Sentiment Analyser v1.0      ║"))
    print(BOLD("  ╚══════════════════════════════════════╝"))

    client = get_client()
    print(GREEN("  [✓] Connected to Azure Text Analytics.\n"))

    if args.file:
        process_file(args.file, client, save_json=args.save)

    elif args.text:
        texts       = [args.text]
        sentiments  = analyse_sentiment(client, texts)
        key_phrases = extract_key_phrases(client, texts)
        languages   = detect_language(client, texts)
        print_result(sentiments[0], key_phrases[0], languages[0], 1)

    else:
        interactive_mode(client)


if __name__ == "__main__":
    main()
