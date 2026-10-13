"""Score saved Baseline and VersionRAG answers with the original LLM rubric.

Reads dataset/evaluation_results.csv and saves scores in
dataset/evaluation_scored.csv by default. Existing scores are skipped on reruns.
"""

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "dataset" / "evaluation_results.csv"
OUTPUT = ROOT / "dataset" / "evaluation_scored.csv"
EVALUATION_LLM = "gpt-4.1"
SOURCE_FIELDS = ["Type", "Question", "Answer", "Response_Baseline", "Response_VersionRAG"]
SCORES = (("Response_Baseline", "Score_Baseline_LLM"),
          ("Response_VersionRAG", "Score_VersionRAG_LLM"))

SYSTEM_PROMPT = """You are a strict but fair evaluation judge for AI-generated answers.
Your task is to compare the model’s answer to the reference answer, and decide whether the model’s answer is factually correct (1) or incorrect (0).
		•	The reference answer represents the factual truth.
	•	The model’s answer must include all key facts and meaning from the reference.
	•	It is acceptable if the model adds extra relevant details, as long as the reference content is accurately and clearly expressed.
	•	The answer must be semantically correct, complete, and not contradict the reference.
	•	Slight rewording is allowed if the truth of the reference answer remains fully preserved.
	•	Do not give credit for partial correctness or vague overlap — the model must clearly and truthfully answer the original question in line with the reference.

	Only output the digit 1 or 0. Do not explain your reasoning.
"""


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


def llm_score(client, question, reference_answer, model_answer):
    if not model_answer.strip():
        raise ValueError("Cannot judge an empty model answer")
    response = client.responses.create(
        model=EVALUATION_LLM,
        instructions=SYSTEM_PROMPT,
        input=(f"Question: {question}\nReference Answer: {reference_answer}\n"
               f"Model Answer: {model_answer}\nScore:"),
        temperature=0,
        max_output_tokens=16,
    )
    score = response.output_text.strip()
    if score not in {"0", "1"}:
        raise ValueError(f"Judge returned {score!r}; no score was saved")
    return score


def judge_csv_file(limit=1, dry_run=False, input_path=INPUT, output_path=OUTPUT):
    rows = load_rows(input_path, output_path)
    if not 1 <= limit <= len(rows):
        raise ValueError(f"--limit must be between 1 and {len(rows)}")
    pending = [(index, response_field, score_field)
               for index, row in enumerate(rows[:limit], start=1)
               for response_field, score_field in SCORES
               if row[score_field] not in {"0", "1"}]
    print(f"LLM judge: {len(pending)} answers pending among the first {limit} questions")
    print(f"Output: {output_path}")
    if dry_run or not pending:
        return

    load_dotenv()
    client = OpenAI()
    for index, response_field, score_field in pending:
        row = rows[index - 1]
        score = llm_score(client, row["Question"], row["Answer"], row[response_field])
        row[score_field] = score
        save_rows(rows, output_path)
        print(f"Q{index:02d} {score_field}: {score}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1,
                        help="Score the first N questions; defaults to 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending count without calling the API or writing files")
    parser.add_argument("--input", type=Path, default=INPUT,
                        help="Answer CSV; defaults to dataset/evaluation_results.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="Scored CSV; defaults to dataset/evaluation_scored.csv")
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    judge_csv_file(args.limit, args.dry_run, input_path, output_path)
