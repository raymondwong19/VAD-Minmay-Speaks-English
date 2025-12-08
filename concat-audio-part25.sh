#!/usr/bin/env bash
set -euo pipefail

# Usage: ./concat-audio-part25.sh [WORKING_DIR]
WD="${1:-.}"
WD="$(cd "$WD" && pwd)"
TMPDIR="$WD/tmp_mix"
mkdir -p "$TMPDIR"

DUB_JSON="$WD/dub.json"
JAP_WAV="$WD/japanese.wav"
DUB_WAV="$WD/dub.wav"
OUTPUT="$WD/output.mp3"

# Find speaker jsons
shopt -s nullglob
SPEAKER_FILES=( "$WD"/speaker*.json )
if [ ${#SPEAKER_FILES[@]} -eq 0 ]; then
  echo "No speaker JSON files found in $WD" >&2
  exit 1
fi

# extract dir_average_score (second field on second line) from CSV speaker file
get_score() {
  awk -F, 'NR==2{print $2}' "$1" 2>/dev/null || echo 0
}

# Determine top speaker across all speaker*.json files
TOP_SPEAKER_ID=""
TOP_SCORE="-1"
for f in "${SPEAKER_FILES[@]}"; do
  base="$(basename "$f")"
  # Extract number from filename; if none, skip
  if [[ "$base" =~ ([0-9]+) ]]; then
    num="${BASH_REMATCH[1]}"
    sid=$(printf "SPEAKER_%02d" "$num")
  else
    # fallback: use filename as identifier (uppercased, non-alnum -> _)
    sid="$(echo "$base" | tr '[:lower:]' '[:upper:]' | sed 's/[^A-Z0-9]/_/g')"
  fi
  score="$(get_score "$f")"
  score="${score:-0}"
  if (( $(echo "$score > $TOP_SCORE" | bc -l) )); then
    TOP_SCORE="$score"
    TOP_SPEAKER_ID="$sid"
  fi
done

if [ -z "$TOP_SPEAKER_ID" ]; then
  echo "Could not determine top speaker." >&2
  exit 1
fi

# Extract segments for top speaker and others
TOP_JSON="$TMPDIR/top_segments.json"
NON_TOP_JSON="$TMPDIR/non_top_segments.json"
jq --arg s "$TOP_SPEAKER_ID" '[.[] | select(.speaker_id==$s)]' "$DUB_JSON" > "$TOP_JSON"
jq --arg s "$TOP_SPEAKER_ID" '[.[] | select(.speaker_id!=$s)]' "$DUB_JSON" > "$NON_TOP_JSON"

# Build timeline (start end source)
TIMELINE="$TMPDIR/timeline.txt"
: > "$TIMELINE"
jq -r '.[] | "\(.start) \(.end) dub"' "$TOP_JSON" >> "$TIMELINE"
jq -r '.[] | "\(.start) \(.end) jpn"' "$NON_TOP_JSON" >> "$TIMELINE"
sort -n -k1,1 "$TIMELINE" -o "$TIMELINE"

# Merge and fill gaps with jpn
MERGED="$TMPDIR/merged_timeline.txt"
: > "$MERGED"
prev_start=""
prev_end=""
prev_source=""
while read -r start end source; do
  if [ -z "$prev_start" ]; then
    if awk "BEGIN{exit !($start>0)}"; then
      printf "0 %s jpn\n" "$start" >> "$MERGED"
    fi
    prev_start="$start"; prev_end="$end"; prev_source="$source"
    continue
  fi

  if awk "BEGIN{exit !($start > $prev_end)}"; then
    printf "%s %s %s\n" "$prev_start" "$prev_end" "$prev_source" >> "$MERGED"
    if awk "BEGIN{exit !($start > $prev_end)}"; then
      printf "%s %s jpn\n" "$prev_end" "$start" >> "$MERGED"
    fi
    prev_start="$start"; prev_end="$end"; prev_source="$source"
  else
    # overlap: extend end and prefer dub if either is dub
    if awk "BEGIN{exit !($end > $prev_end)}"; then prev_end="$end"; fi
    if [ "$source" = "dub" ] || [ "$prev_source" = "dub" ]; then prev_source="dub"; else prev_source="jpn"; fi
  fi
done < "$TIMELINE"

if [ -n "$prev_start" ]; then
  printf "%s %s %s\n" "$prev_start" "$prev_end" "$prev_source" >> "$MERGED"
fi

# Extract pieces and concatenate
i=0
pieces_list="$TMPDIR/pieces.txt"
: > "$pieces_list"
while read -r start end source; do
  out="$TMPDIR/piece_$(printf "%03d" "$i").wav"
  duration=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')
  if [ "$source" = "dub" ]; then
    ffmpeg -hide_banner -loglevel error -y -ss "$start" -t "$duration" -i "$DUB_WAV" -acodec pcm_s16le -ar 44100 -ac 2 "$out"
  else
    ffmpeg -hide_banner -loglevel error -y -ss "$start" -t "$duration" -i "$JAP_WAV" -acodec pcm_s16le -ar 44100 -ac 2 "$out"
  fi
  printf "file '%s'\n" "$out" >> "$pieces_list"
  i=$((i+1))
done < "$MERGED"

CONCAT_WAV="$TMPDIR/concat.wav"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$pieces_list" -c copy "$CONCAT_WAV" || \
  ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$pieces_list" -c:a pcm_s16le "$CONCAT_WAV"

ffmpeg -hide_banner -loglevel error -y -i "$CONCAT_WAV" -b:a 192k "$OUTPUT"

echo "Done. Output: $OUTPUT (top speaker: $TOP_SPEAKER_ID score $TOP_SCORE)"
