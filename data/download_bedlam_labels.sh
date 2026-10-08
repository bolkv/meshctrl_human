#!/usr/bin/env bash
# Download the BEDLAM SMPL-X ground truth (the processed training labels).
#   sfile = bedlam_labels/all_npz_12_training.zip
#   -> per-image SMPL-X parameters (pose/betas/trans, world + camera) for every sequence
#
# Prerequisite: register at https://bedlam.is.tue.mpg.de and accept the licence
# (MPI account, the same one as SMPL-X).
#
# Usage:
#   export BEDLAM_USER='you@example.com'   # same MPI account you used for SMPL-X, if applicable
#   export BEDLAM_PASS='password'
#   bash download_bedlam_labels.sh
set -e

OUT="${1:-/workspace/BEDLAM/bedlam_labels}"
USER="${BEDLAM_USER:?export BEDLAM_USER (your bedlam.is.tue.mpg.de account)}"
PASS="${BEDLAM_PASS:?export BEDLAM_PASS}"
ZIP="all_npz_12_training.zip"

# URL-encode special characters (@ in the email, * in the password, ...) - same as the official script
urle() { local LANG=C i x; for ((i=0;i<${#1};i++)); do x="${1:i:1}";
  [[ "$x" == [a-zA-Z0-9.~-] ]] && printf '%s' "$x" || printf '%%%02X' "'$x"; done; }

mkdir -p "$OUT"; cd "$OUT"

echo ">> downloading SMPL-X labels (download.is.tue.mpg.de, domain=bedlam)..."
wget --post-data "username=$(urle "$USER")&password=$(urle "$PASS")" \
     --no-check-certificate --continue \
     "https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_labels/${ZIP}" \
     -O "$ZIP"

# verify we actually got a zip and not an HTML error page (PK magic bytes)
if [ "$(head -c 2 "$ZIP")" != "PK" ]; then
    echo ""
    echo "FAILED: got HTML instead of a zip ($(wc -c < "$ZIP") bytes)."
    echo "   -> log in at bedlam.is.tue.mpg.de in a browser and check that the BEDLAM licence is accepted."
    echo "      (an SMPL-X account may still need a separate BEDLAM licence agreement)"
    exit 1
fi

echo ">> extracting..."
unzip -q -o "$ZIP" -d ./npz
rm -f "$ZIP"
echo ""
echo "done. SMPL-X labels are in: $OUT/npz"
ls "$OUT/npz" | head
echo "... $(ls "$OUT/npz" | wc -l) files total"
