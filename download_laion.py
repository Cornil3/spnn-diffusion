"""
Download LAION-Aesthetics v2 5+ dataset for SPNN-VAE training.

Steps:
  1. Download parquet metadata from HuggingFace (laion/laion2B-en-aesthetic)
  2. Filter rows: aesthetic >= 5.0, low watermark, safe content
  3. Download & resize images to 256x256 using img2dataset

Usage:
  python download_laion.py --output_dir /path/to/laion-aesthetic-5plus
  python download_laion.py --output_dir /path/to/laion-aesthetic-5plus --skip_download_parquets  # resume from step 2
  python download_laion.py --output_dir /path/to/laion-aesthetic-5plus --skip_filter            # resume from step 3

Requirements:
  pip install img2dataset pyarrow huggingface_hub
"""

import argparse
import os
import glob
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "laion/laion2B-en-aesthetic"


def download_parquets(output_dir):
    """Download all parquet files from HuggingFace."""
    parquet_dir = os.path.join(output_dir, "parquets_raw")
    os.makedirs(parquet_dir, exist_ok=True)

    # List actual parquet files in the repo
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    parquet_files = sorted([f for f in all_files if f.endswith(".parquet")])
    print(f"  Found {len(parquet_files)} parquet files in repo")

    for i, filename in enumerate(parquet_files):
        out_path = os.path.join(parquet_dir, os.path.basename(filename))

        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(parquet_files)}] Already exists: {filename}")
            continue

        print(f"  [{i+1}/{len(parquet_files)}] Downloading {filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=parquet_dir,
        )

    print(f"\nAll parquets saved to {parquet_dir}")
    return parquet_dir


def filter_parquets(parquet_dir, output_dir, min_aesthetic=5.0,
                    max_pwatermark=0.8, max_punsafe=0.5):
    """Filter parquets: aesthetic >= threshold, low watermark, safe content."""
    filtered_dir = os.path.join(output_dir, "parquets_filtered")
    os.makedirs(filtered_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    total_before = 0
    total_after = 0

    for pf in parquet_files:
        out_path = os.path.join(filtered_dir, os.path.basename(pf))
        if os.path.exists(out_path):
            table = pq.read_table(out_path)
            total_after += len(table)
            print(f"  Already filtered: {os.path.basename(pf)} ({len(table)} rows)")
            continue

        table = pq.read_table(pf)
        df = table.to_pandas()
        total_before += len(df)

        filtered = df[
            (df["aesthetic"] >= min_aesthetic) &
            (df["pwatermark"] < max_pwatermark) &
            (df["punsafe"] < max_punsafe)
        ]
        total_after += len(filtered)

        filtered.to_parquet(out_path, index=False)
        print(f"  {os.path.basename(pf)}: {len(df)} -> {len(filtered)} rows")

    print(f"\nFiltering done: {total_after} rows kept")
    print(f"Filtered parquets saved to {filtered_dir}")
    return filtered_dir


def download_images(filtered_dir, output_dir, image_size=256,
                    processes_count=16, thread_count=64,
                    min_image_size=256, max_aspect_ratio=2.0):
    """Download and resize images using img2dataset."""
    from img2dataset import download

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"\nDownloading images to {images_dir}...")
    print(f"  image_size={image_size}, resize_mode=center_crop")
    print(f"  processes={processes_count}, threads={thread_count}")
    print(f"  min_image_size={min_image_size}, max_aspect_ratio={max_aspect_ratio}")

    download(
        url_list=filtered_dir,
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        output_format="webdataset",
        output_folder=images_dir,
        processes_count=processes_count,
        thread_count=thread_count,
        image_size=image_size,
        resize_mode="center_crop",
        resize_only_if_bigger=True,
        min_image_size=min_image_size,
        max_aspect_ratio=max_aspect_ratio,
        encode_quality=95,
        encode_format="jpg",
        retries=3,
        timeout=10,
        number_sample_per_shard=10000,
        save_additional_columns=["aesthetic"],
    )

    print(f"\nDone! Images saved to {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download LAION-Aesthetics v2 5+")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for all downloaded data")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--min_aesthetic", type=float, default=5.0)
    parser.add_argument("--processes_count", type=int, default=16)
    parser.add_argument("--thread_count", type=int, default=64)
    parser.add_argument("--skip_download_parquets", action="store_true",
                        help="Skip step 1 (parquet download)")
    parser.add_argument("--skip_filter", action="store_true",
                        help="Skip steps 1-2 (parquet download + filter)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Download parquets
    parquet_dir = os.path.join(args.output_dir, "parquets_raw")
    if not args.skip_download_parquets and not args.skip_filter:
        print("=" * 60)
        print("Step 1/3: Downloading parquet metadata from HuggingFace")
        print("=" * 60)
        parquet_dir = download_parquets(args.output_dir)

    # Step 2: Filter
    filtered_dir = os.path.join(args.output_dir, "parquets_filtered")
    if not args.skip_filter:
        print("\n" + "=" * 60)
        print("Step 2/3: Filtering parquets (aesthetic >= {})".format(args.min_aesthetic))
        print("=" * 60)
        filtered_dir = filter_parquets(parquet_dir, args.output_dir,
                                       min_aesthetic=args.min_aesthetic)

    # Step 3: Download images
    print("\n" + "=" * 60)
    print("Step 3/3: Downloading and resizing images")
    print("=" * 60)
    download_images(filtered_dir, args.output_dir,
                    image_size=args.image_size,
                    processes_count=args.processes_count,
                    thread_count=args.thread_count)


if __name__ == "__main__":
    main()
