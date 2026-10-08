#!/usr/bin/env bash
# BEDLAM SMPL-X ground truth (처리된 학습 라벨) 다운로드.
#   sfile = bedlam_labels/all_npz_12_training.zip
#   → 전 시퀀스 per-image SMPL-X 파라미터(pose/betas/trans, world+camera) npz
#
# 사전: https://bedlam.is.tue.mpg.de 등록 + 라이선스 동의 (MPI 계정, SMPL-X와 동일).
#
# 사용 (★ 당신 터미널에서):
#   export BEDLAM_USER='가입이메일'      # SMPL-X 받을 때 쓴 MPI 계정과 동일하면 그거
#   export BEDLAM_PASS='비밀번호'
#   bash download_bedlam_labels.sh
set -e

OUT="${1:-/workspace/BEDLAM/bedlam_labels}"
USER="${BEDLAM_USER:?BEDLAM_USER 를 export 하세요 (bedlam.is.tue.mpg.de 계정)}"
PASS="${BEDLAM_PASS:?BEDLAM_PASS 를 export 하세요}"
ZIP="all_npz_12_training.zip"

# 특수문자 URL 인코딩 (이메일 @ , 비번 * 등) — 공식 스크립트와 동일
urle() { local LANG=C i x; for ((i=0;i<${#1};i++)); do x="${1:i:1}";
  [[ "$x" == [a-zA-Z0-9.~-] ]] && printf '%s' "$x" || printf '%%%02X' "'$x"; done; }

mkdir -p "$OUT"; cd "$OUT"

echo ">> SMPL-X 라벨 다운로드 중 (download.is.tue.mpg.de, domain=bedlam)..."
wget --post-data "username=$(urle "$USER")&password=$(urle "$PASS")" \
     --no-check-certificate --continue \
     "https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_labels/${ZIP}" \
     -O "$ZIP"

# 받은 게 진짜 zip인지 검증 (PK)
if [ "$(head -c 2 "$ZIP")" != "PK" ]; then
    echo ""
    echo "❌ 실패: zip이 아니라 HTML 이 내려옴 ($(wc -c < "$ZIP") bytes)."
    echo "   → bedlam.is.tue.mpg.de 에 브라우저 로그인해서 BEDLAM 라이선스 '동의'가 됐는지 확인하세요."
    echo "     (SMPL-X 계정이라도 BEDLAM은 별도 라이선스 동의가 필요할 수 있음)"
    exit 1
fi

echo ">> 압축 해제..."
unzip -q -o "$ZIP" -d ./npz
rm -f "$ZIP"
echo ""
echo "✅ 완료. SMPL-X 라벨 위치: $OUT/npz"
ls "$OUT/npz" | head
echo "... 총 $(ls "$OUT/npz" | wc -l) 개"
