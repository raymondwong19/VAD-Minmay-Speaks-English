#!/usr/bin/env bash
set -euo pipefail

# Usage: ./master.sh /path/to/input.mkv
if [ "${#@}" -ne 1 ]; then
  echo "Usage: $0 /path/to/input.mkv" >&2
  exit 2
fi

MKV="$1"
if [ ! -f "$MKV" ]; then
  echo "File not found: $MKV" >&2
  exit 3
fi

case "$MKV" in
  *.mkv) ;;
  *)
    echo "Input must be an .mkv file" >&2
    exit 4
    ;;
esac

WD="$(pwd)/working-dir"
echo "Working dir: $WD"

# deps quick checks
for cmd in ffmpeg python3 jq; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing required command: $cmd" >&2; exit 5; }
done

# 0. Clean working-dir
rm -rf "$WD"
mkdir -p "$WD"

# 1. Extract audio tracks:
# - Dub (english) -> dub.wav (we assume stream index 1)
# - Japanese -> japanese.wav (we assume stream index 2)
echo "Extracting audio tracks..."
ffmpeg -hide_banner -loglevel error -y -i "$MKV" -map 0:1 -vn -ac 2 -ar 48000 -c:a pcm_s16le "$WD/dub.wav"
ffmpeg -hide_banner -loglevel error -y -i "$MKV" -map 0:2 -vn -ac 2 -ar 48000 -c:a pcm_s16le "$WD/japanese.wav"

# 2. Run diarize_part1.py inside minmay-chroot venv
if [ ! -d "minmay-chroot" ]; then
  echo "minmay-chroot venv not found in CWD; ensure it exists." >&2
  exit 6
fi
if [ ! -f "$WD/dub.wav" ]; then
  echo "Expected $WD/dub.wav missing" >&2
  exit 7
fi

echo "Activating minmay-chroot and running diarize_part1.py..."
# shellcheck disable=SC1090
source minmay-chroot/bin/activate
if ! python3 diarize_part1.py "$WD/dub.wav" 0 "$WD/dub.json"; then
  echo "diarize_part1.py failed" >&2
  deactivate || true
  exit 8
fi
deactivate || true

# 3. Run speaker-splitter-part15.sh on working-dir
if [ ! -x speaker-splitter-part15.sh ]; then
  echo "speaker-splitter-part15.sh not found or not executable in CWD" >&2
  exit 9
fi

echo "Running speaker-splitter-part15.sh $WD ..."
/bin/bash speaker-splitter-part15.sh "$WD"

# 4. For each speaker## in $WD/voices run voiceprint_part2 (inside minmay-embedder venv)
if [ ! -d "$WD/voices" ]; then
  echo "Expected voices directory at $WD/voices not found" >&2
  exit 10
fi

if [ ! -d "minmay-embedder" ]; then
  echo "minmay-embedder venv not found in CWD; ensure it exists." >&2
  exit 11
fi

echo "Activating minmay-embedder and scoring each speaker..."
# shellcheck disable=SC1090
source minmay-embedder/bin/activate

shopt -s nullglob
for spdir in "$WD"/voices/speaker*; do
  [ -d "$spdir" ] || continue
  name="$(basename "$spdir")"
  outjson="$WD/${name}.json"
  echo "Scoring $spdir -> $outjson"
  # The tool usage: python3 compare_dir_embedding.py /path/to/target-dir
  # The user said the internal script name may differ; try common names
  if python3 voiceprint_part2.py "$spdir" | tee "$outjson"; then
    :
  elif python3 compare_dir_embedding.py "$spdir" | tee "$outjson"; then
    :
  else
    echo "Voiceprint script failed for $spdir" >&2
    deactivate || true
    exit 12
  fi
done
deactivate || true
shopt -u nullglob

# 5. Run concat-audio-part25.sh on working-dir
if [ ! -x concat-audio-part25.sh ]; then
  echo "concat-audio-part25.sh not found or not executable in CWD" >&2
  exit 13
fi

echo "Running concat-audio-part25.sh $WD ..."
/bin/bash concat-audio-part25.sh "$WD"

# 6. Final check for output.mp3
if [ -f "$WD/output.mp3" ]; then
  echo "Done. output.mp3 is at: $WD/output.mp3"
  exit 0
else
  echo "concat step completed but output.mp3 not found in $WD" >&2
  exit 14
fi
