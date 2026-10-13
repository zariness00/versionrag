"""Manually score saved Baseline and VersionRAG answers, 1 or 0.

Reads dataset/evaluation_results.csv and saves scores in
dataset/evaluation_scored.csv by default. Existing scores are skipped on reruns.
"""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "dataset" / "evaluation_results.csv"
OUTPUT = ROOT / "dataset" / "evaluation_scored.csv"
SOURCE_FIELDS = ["Type", "Question", "Answer", "Response_Baseline", "Response_VersionRAG"]
SCORES = (("Response_Baseline", "Score_Baseline_Human"),
          ("Response_VersionRAG", "Score_VersionRAG_Human"))


def load_rows(input_path, output_path):
    with input_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != SOURCE_FIELDS:
            raise ValueError(f"Unexpected columns in {input_path}")
        source = list(reader)

    if output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != len(source) or any(
            any(saved.get(field) != original[field] for field in SOURCE_FIELDS)
            for saved, original in zip(rows, source)
        ):
            raise ValueError(f"{output_path} does not match {input_path}")
    else:
        rows = source

    for row in rows:
        for _, score_field in SCORES:
            row.setdefault(score_field, "")
    return rows


def save_rows(rows, output_path):
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)


def judge_csv_file_manually(limit=1, dry_run=False, input_path=INPUT, output_path=OUTPUT):
    rows = load_rows(input_path, output_path)
    if not 1 <= limit <= len(rows):
        raise ValueError(f"--limit must be between 1 and {len(rows)}")
    pending = [(index, response_field, score_field)
               for index, row in enumerate(rows[:limit], start=1)
               for response_field, score_field in SCORES
               if row[score_field] not in {"0", "1"}]
    print(f"Human judge: {len(pending)} answers pending among the first {limit} questions")
    print(f"Output: {output_path}")
    if dry_run or not pending:
        return

    for index, response_field, score_field in pending:
        row = rows[index - 1]
        print(f"\nQ{index:02d}: {row['Question']}")
        print(f"Reference answer: {row['Answer']}")
        print(f"{response_field}: {row[response_field]}")
        while True:
            score = input("Score (1=correct, 0=incorrect, s=skip, q=quit): ").strip().lower()
            if score in {"0", "1", "s", "q"}:
                break
            print("Please enter 1, 0, s, or q.")
        if score == "q":
            print("Stopped. Saved scores remain in the output file.")
            return
        if score == "s":
            continue
        row[score_field] = score
        save_rows(rows, output_path)
        print(f"Saved Q{index:02d} {score_field}: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1,
                        help="Score the first N questions; defaults to 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending count without asking for scores or writing files")
    parser.add_argument("--input", type=Path, default=INPUT,
                        help="Answer CSV; defaults to dataset/evaluation_results.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="Human-scored CSV; defaults to dataset/evaluation_scored.csv")
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    judge_csv_file_manually(args.limit, args.dry_run, input_path, output_path)
