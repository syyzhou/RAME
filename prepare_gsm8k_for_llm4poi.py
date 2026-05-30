#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_rows(path: Path):
    s = str(path).lower()
    if s.endswith(".json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    if s.endswith(".jsonl"):
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if s.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported input format: {path}")


def normalize_row(row):
    if "question" in row and "answer" in row:
        return {"question": str(row["question"]), "answer": str(row["answer"])}
    if "user_prompt" in row and "assistant_prompt" in row:
        return {"question": str(row["user_prompt"]), "answer": str(row["assistant_prompt"])}
    raise ValueError(f"Unsupported row keys: {list(row.keys())}")


def escape_one_line(text: str) -> str:
    return json.dumps(str(text), ensure_ascii=False)[1:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_input", required=True)
    ap.add_argument("--test_input", required=True)
    ap.add_argument("--train_output_json", required=True)
    ap.add_argument("--test_output_txt", required=True)
    args = ap.parse_args()

    train_rows = [normalize_row(r) for r in load_rows(Path(args.train_input))]
    test_rows = [normalize_row(r) for r in load_rows(Path(args.test_input))]

    train_out = Path(args.train_output_json)
    test_out = Path(args.test_output_txt)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    with train_out.open("w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False)

    with test_out.open("w", encoding="utf-8") as f:
        for r in test_rows:
            q = escape_one_line(r["question"])
            a = escape_one_line(r["answer"])
            f.write(f"<question>: {q}<answer>: {a}\n")

    print(f"[ok] train samples: {len(train_rows)} -> {train_out}")
    print(f"[ok] test samples: {len(test_rows)} -> {test_out}")


if __name__ == "__main__":
    main()
