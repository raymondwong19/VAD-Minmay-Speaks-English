#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: ./speaker-splitter-part15.sh /path/to/working-dir (contains dub.wav and dub.json)" >&2
  exit 1
fi

WD="$1"
DUB="$WD/dub.wav"
JSON="$WD/dub.json"
OUTDIR="$WD/voices"
PAD=0.300   # padding in seconds to add to start and end

if [ ! -f "$DUB" ]; then
  echo "dub.wav not found in $WD" >&2
  exit 2
fi
if [ ! -f "$JSON" ]; then
  echo "dub.json not found in $WD" >&2
  exit 3
fi

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "Requires jq. Install it and retry." >&2
  exit 4
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Requires ffmpeg. Install it and retry." >&2
  exit 5
fi

# parse unique speaker ids, normalize to lowercase "speakerNN", and create directories
mapfile -t RAW_SPEAKERS < <(jq -r '.[].speaker_id' "$JSON" | sort -u)
declare -A SP_MAP
for idx in "${!RAW_SPEAKERS[@]}"; do
  raw="${RAW_SPEAKERS[$idx]}"
  if [[ "$raw" =~ ([0-9]+)$ ]]; then
    num=$(printf "%02d" "${BASH_REMATCH[1]}")
  else
    num=$(printf "%02d" "$idx")
  fi
  norm="speaker${num}"
  SP_MAP["$raw"]="$norm"
  mkdir -p "$OUTDIR/$norm"
done

# get dub.wav duration to clamp padding
DUB_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$DUB")
DUB_DUR=$(awk "BEGIN{printf \"%.3f\", ${DUB_DUR:-0}}")

# iterate turns and cut with ffmpeg (preserve sample rate/channels; force stereo 44.1k pcm_s16le)
declare -A COUNTS
jq -c '.[]' "$JSON" | while IFS= read -r turn; do
  start=$(jq -r '.start' <<<"$turn")
  end=$(jq -r '.end' <<<"$turn")
  rawsp=$(jq -r '.speaker_id' <<<"$turn")
  sp="${SP_MAP[$rawsp]}"
  # apply padding, clamped to file bounds
  start_p=$(awk -v s="$start" -v p="$PAD" 'BEGIN{t=s-p; if(t<0) t=0; printf "%.3f", t}')
  end_p=$(awk -v e="$end" -v p="$PAD" -v md="$DUB_DUR" 'BEGIN{t=e+p; if(t>md) t=md; printf "%.3f", t}')
  duration=$(awk "BEGIN{printf \"%.3f\", ($end_p - $start_p)}")
  c=${COUNTS[$sp]:-0}
  c=$((c+1))
  COUNTS[$sp]=$c
  sfile=$(printf "seg%02d.wav" "$c")
  outpath="$OUTDIR/$sp/$sfile"
  ffmpeg -hide_banner -loglevel error -nostdin -y -i "$DUB" -ss "$start_p" -t "$duration" -vn -ac 2 -ar 44100 -c:a pcm_s16le "$outpath"
  echo "WROTE $outpath"
done

echo "Done. Speaker dirs under: $OUTDIR"
