import json
import re
from pathlib import Path


SRC_TRAIN = Path("datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.json")
SRC_TEST = Path("datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.txt")
OUT_TRAIN = Path("datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0_top10.json")
OUT_TEST = Path("datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0_top10.txt")
OUT_STATS = Path("datasets/NYC/preprocessed/gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0_top10_stats.json")


def load_test_txt(path):
    text = path.read_text(encoding="utf-8")
    pat = re.compile(r"<question>:\s*(.*?)\s*<answer>:\s*(.*?)(?=\n<question>:|\Z)", re.S)
    return [{"question": m.group(1).strip(), "answer": m.group(2).strip()} for m in pat.finditer(text)]


def truncate_question(question, k=10):
    def repl(match):
        body = match.group(1)
        kept = []
        cand_seen = 0
        for line in body.splitlines():
            if re.match(r"\s*\d+\s*\|", line):
                cand_seen += 1
                if cand_seen <= k:
                    kept.append(line)
            else:
                kept.append(line)
        return "<candidates>" + "\n".join(kept).rstrip() + "\n</candidates>"

    return re.sub(r"<candidates>(.*?)</candidates>", repl, question, flags=re.S)


def calc(rows):
    total = hit1 = hit5 = hit10 = hit20 = missing = 0
    for row in rows:
        question = row.get("question", "").replace("\\\\n", "\n")
        answer = str(row.get("answer", ""))
        nums = re.findall(r"\d+", answer)
        if not nums:
            continue
        target = int(nums[0])
        block = re.search(r"<candidates>\s*(.*?)\s*</candidates>", question, re.S)
        if not block:
            missing += 1
            continue
        ids = []
        for line in block.group(1).splitlines():
            match = re.match(r"\s*(\d+)\s*\|", line)
            if match:
                ids.append(int(match.group(1)))
        total += 1
        hit1 += target in ids[:1]
        hit5 += target in ids[:5]
        hit10 += target in ids[:10]
        hit20 += target in ids[:20]
    return {
        "samples": total,
        "rewritten": total - missing,
        "avg_candidates": 10.0,
        "target_in_candidates@1": hit1 / max(total, 1),
        "target_in_candidates@5": hit5 / max(total, 1),
        "target_in_candidates@10": hit10 / max(total, 1),
        "target_in_candidates@20": hit20 / max(total, 1),
        "missing_candidate_block": missing,
    }


def main():
    train_rows = json.loads(SRC_TRAIN.read_text(encoding="utf-8"))
    test_rows = load_test_txt(SRC_TEST)
    train_top10 = [{"question": truncate_question(r["question"], 10), "answer": r["answer"]} for r in train_rows]
    test_top10 = [{"question": truncate_question(r["question"], 10), "answer": r["answer"]} for r in test_rows]

    OUT_TRAIN.write_text(json.dumps(train_top10, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_TEST.write_text(
        "\n".join(f"<question>: {r['question']} <answer>: {r['answer']}" for r in test_top10) + "\n",
        encoding="utf-8",
    )

    stats = {
        "artifact_dir": "rag/target_poi_multiview/gsm8k_final",
        "strategy": "truncate reranked top20 to top10; original mix=retriever10+transition4+geo3+history3; rerank weights=(retriever=0.75, transition=0.75, history=1.0, geo=0.0)",
        "train": calc(train_top10),
        "test": calc(test_top10),
        "outputs": {
            "train": str(OUT_TRAIN),
            "test": str(OUT_TEST),
            "stats": str(OUT_STATS),
        },
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
