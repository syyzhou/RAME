import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def build_region_ids(poi_base: pd.DataFrame, lat_bins: int, lon_bins: int) -> dict:
    n_pois = len(poi_base)
    lat_rank = poi_base["Latitude"].rank(method="first").astype(int) - 1
    lon_rank = poi_base["Longitude"].rank(method="first").astype(int) - 1
    lat_bin = (lat_rank * lat_bins // n_pois).clip(0, lat_bins - 1).astype(int)
    lon_bin = (lon_rank * lon_bins // n_pois).clip(0, lon_bins - 1).astype(int)
    region_id = lat_bin * lon_bins + lon_bin
    return dict(zip(poi_base["PoiId"], region_id.astype(int)))


def build_transfer_neighbors(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    prev_neighbors = defaultdict(set)
    next_neighbors = defaultdict(set)
    sort_cols = ["UserId", "pseudo_session_trajectory_id", "UTCTimeOffset"]

    for _, group in df.sort_values(sort_cols).groupby(
        ["UserId", "pseudo_session_trajectory_id"], sort=False
    ):
        pois = group["PoiId"].tolist()
        for prev_poi, next_poi in zip(pois, pois[1:]):
            prev_poi = int(prev_poi)
            next_poi = int(next_poi)
            next_neighbors[prev_poi].add(next_poi)
            prev_neighbors[next_poi].add(prev_poi)

    return prev_neighbors, next_neighbors


def build_rqvae_poi_info(
    train_csv: Path,
    output_csv: Path,
    lat_bins: int,
    lon_bins: int,
) -> pd.DataFrame:
    required_cols = [
        "UserId",
        "PoiId",
        "PoiCategoryId",
        "Latitude",
        "Longitude",
        "UTCTimeOffset",
        "pseudo_session_trajectory_id",
    ]
    df = pd.read_csv(train_csv, usecols=required_cols)
    df["UTCTimeOffset"] = pd.to_datetime(df["UTCTimeOffset"], errors="coerce")
    df = df.dropna(subset=["UTCTimeOffset"]).copy()
    df["Hour"] = df["UTCTimeOffset"].dt.hour.astype(int)

    poi_base = (
        df.sort_values("UTCTimeOffset")
        .groupby("PoiId", as_index=False)
        .agg(Latitude=("Latitude", "first"), Longitude=("Longitude", "first"))
    )
    region_map = build_region_ids(poi_base, lat_bins, lon_bins)
    prev_neighbors, next_neighbors = build_transfer_neighbors(df)

    records = []
    for pid, group in df.groupby("PoiId", sort=True):
        pid = int(pid)
        records.append(
            {
                "Pid": pid,
                "Uid": sorted(int(x) for x in group["UserId"].dropna().unique()),
                "Catname": sorted(int(x) for x in group["PoiCategoryId"].dropna().unique()),
                "Region": [int(region_map[pid])],
                "Time": sorted(int(x) for x in group["Hour"].dropna().unique()),
                "neighbors": sorted(prev_neighbors.get(pid, set())),
                "forward_neighbors": sorted(next_neighbors.get(pid, set())),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the GNPR-SID V1 RQ-VAE POI feature table from a split train_sample.csv."
    )
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--lat_bins", type=int, default=31)
    parser.add_argument("--lon_bins", type=int, default=30)
    args = parser.parse_args()

    out_df = build_rqvae_poi_info(
        train_csv=args.train_csv,
        output_csv=args.output_csv,
        lat_bins=args.lat_bins,
        lon_bins=args.lon_bins,
    )
    print(f"saved: {args.output_csv}")
    print(f"pois: {len(out_df)}")


if __name__ == "__main__":
    main()
