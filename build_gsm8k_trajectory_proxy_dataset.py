#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


ENTRY_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(\d+)\s*\|\s*([^\n<]+)"
)
USER_RE = re.compile(r"user\s+(\d+)", re.IGNORECASE)
QUERY_TIME_RE = re.compile(
    r"Given the data,\s*at\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
    re.IGNORECASE,
)
TXT_QA_RE = re.compile(
    r"<question>:\s*(.*?)\s*<answer>:\s*(.*?)\s*(?=(<question>:|$))",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class QASample:
    question: str
    answer: str


def load_qa(path: Path) -> List[QASample]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [QASample(question=str(x["question"]), answer=str(x["answer"])) for x in data]
    if path.suffix.lower() == ".txt":
        rows: List[QASample] = []
        text = path.read_text(encoding="utf-8").strip()
        for m in TXT_QA_RE.finditer(text):
            rows.append(QASample(question=m.group(1).strip(), answer=m.group(2).strip()))
        if not rows and text:
            raise ValueError(f"Cannot parse txt QA file: {str(path)}")
        return rows
    raise ValueError(f"Unsupported QA format: {path}")


def parse_question(question: str) -> Tuple[int, str, List[Tuple[str, int, str]]]:
    user_m = USER_RE.search(question)
    if not user_m:
        raise ValueError(f"Cannot find user id from question: {question[:200]}...")
    user_id = int(user_m.group(1))

    query_time_m = QUERY_TIME_RE.search(question)
    if not query_time_m:
        raise ValueError(f"Cannot find query time from question: {question[:200]}...")
    query_time = query_time_m.group(1)

    entries = []
    for t, poi, cat in ENTRY_RE.findall(question):
        entries.append((t, int(poi), cat.strip()))
    if not entries:
        raise ValueError(f"No (time|poi|category) entries found: {question[:200]}...")
    return user_id, query_time, entries


def build_user_index(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    if "UserId" not in df.columns or "UTCTimeOffsetEpoch" not in df.columns:
        raise ValueError("Raw csv missing required columns: UserId / UTCTimeOffsetEpoch")
    df = df.copy()
    df["UserId"] = df["UserId"].astype(int)
    return {int(uid): g.sort_values("UTCTimeOffsetEpoch").reset_index(drop=True) for uid, g in df.groupby("UserId")}


def find_anchor_row(user_df: pd.DataFrame, earliest_t: str, earliest_poi: int) -> Optional[pd.Series]:
    m = user_df[
        (user_df["UTCTimeOffset"].astype(str) == earliest_t)
        & (user_df["PoiId"].astype(int) == int(earliest_poi))
    ]
    if len(m) > 0:
        return m.iloc[0]

    # fallback: time-only
    m2 = user_df[user_df["UTCTimeOffset"].astype(str) == earliest_t]
    if len(m2) > 0:
        return m2.iloc[0]
    return None


def find_current_row(
    user_df: pd.DataFrame, entries: List[Tuple[str, int, str]]
) -> Optional[pd.Series]:
    # "当前轨迹"在 GSM8K 模板里对应 <current> 的末尾记录，通常也是 entries 中最晚时间点。
    latest_t, latest_poi, _ = max(entries, key=lambda x: x[0])
    m = user_df[
        (user_df["UTCTimeOffset"].astype(str) == latest_t)
        & (user_df["PoiId"].astype(int) == int(latest_poi))
    ]
    if len(m) > 0:
        return m.iloc[0]
    # fallback: time-only
    m2 = user_df[user_df["UTCTimeOffset"].astype(str) == latest_t]
    if len(m2) > 0:
        return m2.iloc[0]
    return None


def rows_from_anchor_to_current_trajectory(
    user_df: pd.DataFrame, anchor_row: pd.Series, current_row: pd.Series
) -> pd.DataFrame:
    traj_col = "pseudo_session_trajectory_id"
    if traj_col not in user_df.columns:
        raise ValueError(f"Raw csv missing required column: {traj_col}")

    traj_first_epoch = (
        user_df.groupby(traj_col, sort=False)["UTCTimeOffsetEpoch"].min().sort_values()
    )
    anchor_traj = anchor_row[traj_col]
    current_traj = current_row[traj_col]
    anchor_start = traj_first_epoch.loc[anchor_traj]
    current_start = traj_first_epoch.loc[current_traj]
    low = min(anchor_start, current_start)
    high = max(anchor_start, current_start)
    keep_traj = traj_first_epoch[(traj_first_epoch >= low) & (traj_first_epoch <= high)].index

    out = user_df[user_df[traj_col].isin(keep_traj)].copy()
    out = out.sort_values("UTCTimeOffsetEpoch").reset_index(drop=True)
    return out


def filter_rows_before_query_time(rows: pd.DataFrame, query_time: str) -> pd.DataFrame:
    query_dt = pd.to_datetime(query_time)
    row_times = pd.to_datetime(rows["UTCTimeOffset"])
    return rows[row_times < query_dt].copy()


def build_proxy_question(user_id: int, query_time: str, rows: pd.DataFrame, poi_upper: int) -> str:
    parts = [
        f"<question>: The following data contains check-in sequences of user {user_id}:",
        "[Current trajectory's check-in sequence]:",
    ]
    for _, r in rows.iterrows():
        parts.append(
            f"At {r['UTCTimeOffset']}, user {user_id} visited POI id {int(r['PoiId'])} "
            f"which is a {r['PoiCategoryName']} with Category id {int(r['PoiCategoryId'])}."
        )
    parts.append(
        f"Given the data, At {query_time}, Which POI id will user {user_id} visit? "
        f"Note that POI id is an integer in the range from 0 to {poi_upper}."
    )
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_input", required=True, help="GSM8K-style QA file (.json/.txt)")
    ap.add_argument("--raw_csv", required=True, help="Raw trajectory csv (train_sample.csv / test_sample_with_traj.csv)")
    ap.add_argument("--output_json", required=True, help="Proxy QA json for precompute_trajectory_embeddings.py")
    ap.add_argument("--poi_upper", type=int, default=4981)
    args = ap.parse_args()

    qa_samples = load_qa(Path(args.qa_input))
    raw_df = pd.read_csv(args.raw_csv)
    user_index = build_user_index(raw_df)

    out = []
    fail_count = 0
    for i, s in enumerate(qa_samples):
        try:
            user_id, query_time, entries = parse_question(s.question)
            earliest_t, earliest_poi, _ = min(entries, key=lambda x: x[0])
            user_df = user_index.get(user_id)
            if user_df is None or len(user_df) == 0:
                raise ValueError(f"user {user_id} not found in raw csv")
            anchor = find_anchor_row(user_df, earliest_t, earliest_poi)
            if anchor is None:
                raise ValueError(f"anchor not found: user={user_id}, t={earliest_t}, poi={earliest_poi}")
            current = find_current_row(user_df, entries)
            if current is None:
                raise ValueError(f"current row not found: user={user_id}")
            rows = rows_from_anchor_to_current_trajectory(user_df, anchor, current)
            rows = filter_rows_before_query_time(rows, query_time)
            if rows.empty:
                raise ValueError(f"no context rows before query time: user={user_id}, query_time={query_time}")
            q_proxy = build_proxy_question(user_id, query_time, rows, args.poi_upper)
            out.append({"question": q_proxy, "answer": s.answer})
        except Exception as e:
            fail_count += 1
            # fallback to a minimally compatible proxy to preserve index alignment
            out.append({
                "question": (
                    "<question>: The following data contains check-in sequences of user 0: "
                    "[Current trajectory's check-in sequence]: "
                    "Given the data, At 1970-01-01 00:00:00, Which POI id will user 0 visit? "
                    f"Note that POI id is an integer in the range from 0 to {args.poi_upper}."
                ),
                "answer": s.answer,
            })
            if fail_count <= 10:
                print(f"[warn] sample {i} fallback due to: {e}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] output: {args.output_json}")
    print(f"[ok] samples: {len(out)}")
    print(f"[ok] fallback samples: {fail_count}")


if __name__ == "__main__":
    main()
