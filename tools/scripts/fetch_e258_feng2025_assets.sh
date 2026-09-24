#!/usr/bin/env bash
# Official Figshare assets only. No expression file is opened during download.
set -euo pipefail

dest="${1:-/home/yyf/data/feng2025_candidate}"
mkdir -p "$dest"

fetch_one() {
  local id="$1" name="$2" expected_md5="$3" expected_size="$4"
  local path="$dest/$name" part="$dest/$name.part" actual size attempt
  if [[ -f "$path" ]]; then
    actual="$(md5sum "$path" | awk '{print $1}')"
    size="$(stat -c '%s' "$path")"
    if [[ "$actual" == "$expected_md5" && "$size" == "$expected_size" ]]; then
      printf 'VERIFIED %s %s bytes\n' "$name" "$size"
      return
    fi
    printf 'Existing file failed checksum: %s\n' "$path" >&2
    return 1
  fi
  for attempt in {1..20}; do
    printf 'FETCH %s attempt %s\n' "$name" "$attempt"
    if curl --fail --location --retry 3 --retry-all-errors \
      --connect-timeout 30 --speed-limit 1024 --speed-time 180 \
      --continue-at - --output "$part" \
      "https://ndownloader.figshare.com/files/$id"; then
      size="$(stat -c '%s' "$part")"
      actual="$(md5sum "$part" | awk '{print $1}')"
      if [[ "$size" == "$expected_size" && "$actual" == "$expected_md5" ]]; then
        mv "$part" "$path"
        printf 'VERIFIED %s %s bytes\n' "$name" "$size"
        return
      fi
      printf 'Checksum/size failed for %s: %s %s\n' "$name" "$actual" "$size" >&2
      return 1
    fi
    sleep 10
  done
  printf 'Download attempts exhausted: %s\n' "$name" >&2
  return 1
}

fetch_one 51051809 TargetedScreen_Cell-Metadata.tsv.gz 980a440a8132a20567c4c3610e6a5b20 13491635
fetch_one 49291675 TargetedScreen_LFC_byGene-perLine.tsv.gz 27ae8d0109b9c42f1f972c45987db77e 1919661029
fetch_one 51072902 TargetedScreen_RNA-UMI-Counts.csv.gz f0d613b1ea8147425501baddba034158 4480969391
printf 'E258 official data assets downloaded and checksum-verified. No test truth has been evaluated.\n'
