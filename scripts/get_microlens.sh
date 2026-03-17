#!/usr/bin/env bash

set -euo pipefail

subset="100k"
dest_dir="data/raw/microlens"
include_features=1
include_covers=0
include_frames=0
include_videos=0
include_audio=0
extract_archives=1

usage() {
  cat <<'EOF'
Usage: scripts/get_microlens.sh [options]

Download the official MicroLens dataset files into this repo.

Defaults:
  - subset: 100k
  - destination: data/raw/microlens
  - downloads: core metadata/interactions + extracted modality features
  - does not download covers, raw videos, raw audio, or frame archives unless requested

Options:
  --subset {50k|100k}      Dataset split to download. Default: 100k
  --dest DIR               Target directory. Default: data/raw/microlens
  --core-only              Download only the core tabular/text files
  --with-features          Download extracted modality features when available
  --with-covers            Download cover archive
  --with-frames            Download frame archive (~18-19 GB)
  --with-videos            Mirror raw video directory
  --with-audio             Mirror raw audio directory
  --no-extract             Keep zip files compressed
  -h, --help               Show this help text

Examples:
  scripts/get_microlens.sh
  scripts/get_microlens.sh --subset 50k --core-only
  scripts/get_microlens.sh --subset 100k --with-covers --with-videos
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)
      subset="${2:-}"
      shift 2
      ;;
    --dest)
      dest_dir="${2:-}"
      shift 2
      ;;
    --core-only)
      include_features=0
      shift
      ;;
    --with-features)
      include_features=1
      shift
      ;;
    --with-covers)
      include_covers=1
      shift
      ;;
    --with-frames)
      include_frames=1
      shift
      ;;
    --with-videos)
      include_videos=1
      shift
      ;;
    --with-audio)
      include_audio=1
      shift
      ;;
    --no-extract)
      extract_archives=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$subset" != "50k" && "$subset" != "100k" ]]; then
  echo "--subset must be 50k or 100k" >&2
  exit 1
fi

if [[ "$subset" == "50k" && "$include_features" -eq 1 ]]; then
  echo "Extracted modality features are only published for MicroLens-100k. Disabling --with-features." >&2
  include_features=0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not installed." >&2
  exit 1
fi

if [[ "$extract_archives" -eq 1 ]] && ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required for archive extraction. Re-run with --no-extract or install unzip." >&2
  exit 1
fi

base_url="https://recsys.westlake.edu.cn/MicroLens-${subset}-Dataset"
root_dir="${dest_dir}/MicroLens-${subset}"

mkdir -p "$root_dir"

download_file() {
  local url="$1"
  local output="$2"

  if [[ -f "$output" ]]; then
    echo "Skipping existing file: $output"
    return
  fi

  echo "Downloading $url"
  curl -fL --retry 5 --retry-delay 2 --continue-at - -o "$output" "$url"
}

download_and_extract_zip() {
  local url="$1"
  local zip_path="$2"
  local extract_to="$3"

  download_file "$url" "$zip_path"

  if [[ "$extract_archives" -eq 1 ]]; then
    mkdir -p "$extract_to"
    echo "Extracting $zip_path -> $extract_to"
    unzip -oq "$zip_path" -d "$extract_to"
  fi
}

mirror_directory_listing() {
  local remote_dir="$1"
  local local_dir="$2"

  mkdir -p "$local_dir"

  echo "Mirroring index from $remote_dir"
  mapfile -t entries < <(
    curl -fsSL "$remote_dir/" \
      | grep -oE 'href="[^"]+"' \
      | sed 's/^href="//; s/"$//' \
      | grep -vE '^\.\./?$'
  )

  for entry in "${entries[@]}"; do
    local clean_entry="${entry%/}"
    if [[ -z "$clean_entry" ]]; then
      continue
    fi
    download_file "${remote_dir}/${clean_entry}" "${local_dir}/${clean_entry}"
  done
}

common_files=(
  "readme.txt"
  "extract_audio.py"
  "MicroLens-${subset}_pairs.csv"
  "MicroLens-${subset}_pairs.tsv"
  "MicroLens-${subset}_likes_and_views.txt"
)

if [[ "$subset" == "50k" ]]; then
  common_files+=("MicroLens-50k_titles.csv")
else
  common_files+=(
    "MicroLens-100k_title_en.csv"
    "MicroLens-100k_comment_en.txt"
    "tags_to_summary.csv"
  )
fi

for file_name in "${common_files[@]}"; do
  download_file "${base_url}/${file_name}" "${root_dir}/${file_name}"
done

if [[ "$include_features" -eq 1 ]]; then
  feature_dir="${root_dir}/extracted_modality_features"
  mkdir -p "$feature_dir"

  feature_files=(
    "MicroLens-100k_image_features_CLIPRN50.npy"
    "MicroLens-100k_title_en_text_features_BgeM3.npy"
    "MicroLens-100k_video_features_VideoMAE.npy"
  )

  for file_name in "${feature_files[@]}"; do
    download_file "${base_url}/extracted_modality_features/${file_name}" "${feature_dir}/${file_name}"
  done
fi

if [[ "$include_covers" -eq 1 ]]; then
  download_and_extract_zip \
    "${base_url}/MicroLens-${subset}_covers.zip" \
    "${root_dir}/MicroLens-${subset}_covers.zip" \
    "${root_dir}"
fi

if [[ "$include_frames" -eq 1 ]]; then
  download_and_extract_zip \
    "${base_url}/MicroLens-${subset}_frames_interval_1_number_5.zip" \
    "${root_dir}/MicroLens-${subset}_frames_interval_1_number_5.zip" \
    "${root_dir}"
fi

if [[ "$include_videos" -eq 1 ]]; then
  mirror_directory_listing \
    "${base_url}/MicroLens-${subset}_videos" \
    "${root_dir}/MicroLens-${subset}_videos"
fi

if [[ "$include_audio" -eq 1 ]]; then
  mirror_directory_listing \
    "${base_url}/MicroLens-${subset}_audio" \
    "${root_dir}/MicroLens-${subset}_audio"
fi

echo "MicroLens download complete in ${root_dir}"
