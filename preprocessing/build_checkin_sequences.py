import argparse
from pathlib import Path

import pandas as pd


SPLIT_FILES = {
    "train": "train_sample.csv",
    "validation": "validate_sample_with_traj.csv",
    "test": "test_sample_with_traj.csv",
}


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / SPLIT_FILES[split]
    df = pd.read_csv(path)
    df["SplitTag"] = split
    return df


def build_sequences(data_dir: Path, max_checkins: int, min_train_checkins: int) -> pd.DataFrame:
    frames = [load_split(data_dir, split) for split in SPLIT_FILES]
    df = pd.concat(frames, ignore_index=True)
    df["UTCTimeOffset"] = pd.to_datetime(df["UTCTimeOffset"], errors="coerce")
    df = df.dropna(subset=["UTCTimeOffset"])

    train_df = df[df["SplitTag"] == "train"]
    train_users = set(train_df["UserId"].unique())
    train_pois = set(train_df["PoiId"].unique())
    df = df[df["UserId"].isin(train_users) & df["PoiId"].isin(train_pois)].copy()
    df = df.sort_values(["UserId", "UTCTimeOffset"]).reset_index(drop=True)

    records = []
    for user_id, user_df in df.groupby("UserId", sort=False):
        user_df = user_df.sort_values("UTCTimeOffset").reset_index(drop=True)

        for (traj_id, split_tag), traj_df in user_df.groupby(
            ["pseudo_session_trajectory_id", "SplitTag"], sort=False
        ):
            start_pos = int(traj_df.index.min())
            history_count = start_pos
            history_tail = user_df.iloc[:start_pos].tail(max_checkins)
            traj_df = traj_df.sort_values("UTCTimeOffset").reset_index(drop=True)
            current_df = traj_df.copy()
            merged_df = pd.concat([history_tail, current_df], axis=0).sort_values("UTCTimeOffset")
            merged_df = merged_df.tail(max_checkins).reset_index(drop=True)

            if split_tag == "train" and len(merged_df) < min_train_checkins:
                continue

            records.append(
                {
                    "UserId": user_id,
                    "trajectory_id": traj_id,
                    "SplitTag": split_tag,
                    "history_count": history_count,
                    "current_count": len(current_df),
                    "final_count": len(merged_df),
                    "sequence_PoiId": merged_df["PoiId"].tolist(),
                    "sequence_PoiCategoryId": merged_df["PoiCategoryId"].tolist(),
                    "sequence_PoiCategoryName": merged_df["PoiCategoryName"].tolist(),
                    "sequence_Latitude": merged_df["Latitude"].tolist(),
                    "sequence_Longitude": merged_df["Longitude"].tolist(),
                    "sequence_UTCTimeOffset": merged_df["UTCTimeOffset"].astype(str).tolist(),
                    "sequence_pseudo_session_trajectory_id": merged_df[
                        "pseudo_session_trajectory_id"
                    ].tolist(),
                }
            )

    return pd.DataFrame(records)


def save_outputs(result_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / "trajectory_sequence_data.csv", index=False)

    simple_cols = ["UserId", "sequence_PoiId", "sequence_UTCTimeOffset"]
    split_outputs = {
        "train": "train_poi_sequence.csv",
        "validation": "validation_poi_sequence.csv",
        "test": "test_poi_sequence.csv",
    }
    for split, filename in split_outputs.items():
        split_df = result_df[result_df["SplitTag"] == split][simple_cols]
        split_df.to_csv(output_dir / filename, index=False)
        print(f"{split}: {len(split_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GNPR-SID style max-N check-in sequence files from LLM4POI split CSVs."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("LLM4POI/datasets/ca/preprocessed"),
        help="Directory containing train_sample.csv, validate_sample_with_traj.csv, and test_sample_with_traj.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to DATA_DIR/sequence_50 for max_checkins=50.",
    )
    parser.add_argument("--max_checkins", type=int, default=50)
    parser.add_argument("--min_train_checkins", type=int, default=20)
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir / f"sequence_{args.max_checkins}"
    result_df = build_sequences(args.data_dir, args.max_checkins, args.min_train_checkins)
    save_outputs(result_df, output_dir)
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
