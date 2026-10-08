# Prerequisite: register at https://smpl-x.is.tue.mpg.de, confirm your email, accept the licence.
#
#   export SMPLX_USER='you@example.com'
#   export SMPLX_PASS='password'
#   bash download_smplx.sh            # -> ./models
#   bash download_smplx.sh <out_dir>

set -e

OUT_DIR="${1:-./models}"
USER="${SMPLX_USER:?export SMPLX_USER (the email you registered with)}"
PASS="${SMPLX_PASS:?export SMPLX_PASS (your password)}"
ZIP="models_smplx_v1_1.zip"

urle() { python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$1"; }

mkdir -p "$OUT_DIR"; cd "$OUT_DIR"

echo ">> downloading (POSTing credentials straight to download.php)..."
wget --post-data "username=$(urle "$USER")&password=$(urle "$PASS")" \
     --no-check-certificate --continue \
     "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=${ZIP}&resume=1" \
     -O "$ZIP"

# verify we actually got a zip and not an HTML error page (PK magic bytes)
if [ "$(head -c 2 "$ZIP")" != "PK" ]; then
    echo ""
    echo "FAILED: got HTML instead of a zip ($(wc -c < "$ZIP") bytes)."
    echo "   likely causes:"
    echo "   1) wrong email/password, or the email confirmation was never completed"
    echo "   2) the SMPL-X licence has not been accepted on the website"
    echo "   -> log in at https://smpl-x.is.tue.mpg.de in a browser and check that Download works"
    exit 1
fi

echo ">> extracting..."
unzip -o "$ZIP"
# the archive nests them as models/smplx/... -> flatten to <out_dir>/smplx/...
if [ -d models/smplx ]; then rm -rf ./smplx; mv models/smplx ./smplx; rm -rf models; fi

rm -f "$ZIP"
echo ""
echo "done:"
ls -la ./smplx/SMPLX_*.npz
