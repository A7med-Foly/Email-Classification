"""
predict_cli.py — Classify emails as spam or ham from the command line.

Usage
-----
Single email:
    python predict_cli.py --text "Congratulations! You've won a free iPhone!"

Multiple emails from a text file (one per line):
    python predict_cli.py --file emails.txt

Interactive mode (no flags):
    python predict_cli.py
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BEST_MODEL_PATH, LABEL_ENCODER_PATH, VECTORIZER_PATH
from src.logger import setup_logger
from src.predict import predict_from_paths
from src.text_preprocessing import download_nltk_resources


def parse_args():
    p = argparse.ArgumentParser(description="Email Spam Classifier")
    p.add_argument("--text",  type=str, help="Email text to classify")
    p.add_argument("--file",  type=str, help="Path to a .txt file (one email per line)")
    p.add_argument("--model", type=str, default=BEST_MODEL_PATH)
    return p.parse_args()


def print_result(result: dict) -> None:
    label = result["label"].upper()
    conf  = result["confidence"]
    icon  = "🚨 SPAM" if result["label"] == "spam" else "✅ HAM"
    conf_str = f"  (confidence: {conf:.1%})" if conf is not None else ""
    preview = result["text"][:80] + ("…" if len(result["text"]) > 80 else "")
    print(f"\n  {icon}{conf_str}")
    print(f"  Text: \"{preview}\"")
    print("  " + "─" * 55)


def main():
    setup_logger()
    download_nltk_resources()
    args = parse_args()

    if args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        print("\n── Email Spam Classifier ──")
        print("Type an email and press Enter (blank line to quit):\n")
        texts = []
        while True:
            line = input("  > ").strip()
            if not line:
                break
            texts.append(line)

    if not texts:
        print("No input provided.")
        return

    results = predict_from_paths(texts, VECTORIZER_PATH, BEST_MODEL_PATH, LABEL_ENCODER_PATH)
    print(f"\n{'═'*57}")
    print(f"  RESULTS — {len(results)} email(s) classified")
    print(f"{'═'*57}")
    for r in results:
        print_result(r)


if __name__ == "__main__":
    main()
