#!/usr/bin/env bash
# SMPL-X 모델 다운로드 (MPI 표준 방식)
#
# 사전: https://smpl-x.is.tue.mpg.de 가입 + 이메일 인증 + 라이선스 동의
#
# 사용법 (★ 당신 터미널에서 실행하세요. 제 쉘은 당신 export를 못 봅니다):
#   export SMPLX_USER='bolkv1101@kaist.ac.kr'
#   export SMPLX_PASS='hj020811**'
#   bash download_smplx.sh            # 출력 ./models
#   bash download_smplx.sh /경로      # 출력 경로 지정
#
# 결과: <out>/smplx/SMPLX_NEUTRAL.npz (+ MALE / FEMALE)
set -e

OUT_DIR="${1:-./models}"
USER="${SMPLX_USER:?SMPLX_USER 를 export 하세요 (가입 이메일)}"
PASS="${SMPLX_PASS:?SMPLX_PASS 를 export 하세요 (비밀번호)}"
ZIP="models_smplx_v1_1.zip"

# 이메일의 @ , 비번의 * 등 특수문자 URL-인코딩 (핵심!)
urle() { python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$1"; }

mkdir -p "$OUT_DIR"; cd "$OUT_DIR"

echo ">> 다운로드 중 (아이디/비번을 download.php 로 직접 POST)..."
wget --post-data "username=$(urle "$USER")&password=$(urle "$PASS")" \
     --no-check-certificate --continue \
     "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=${ZIP}&resume=1" \
     -O "$ZIP"

# 받은 게 진짜 zip인지 검증 (PK 매직바이트)
if [ "$(head -c 2 "$ZIP")" != "PK" ]; then
    echo ""
    echo "❌ 실패: zip이 아니라 HTML 이 내려왔습니다 ($(wc -c < "$ZIP") bytes)."
    echo "   원인 후보:"
    echo "   1) 이메일/비번 오타 또는 이메일 인증(confirm) 미완료"
    echo "   2) 웹사이트에서 SMPL-X 라이선스 '동의' 안 함"
    echo "   → 브라우저로 https://smpl-x.is.tue.mpg.de 로그인해서 Download 가 되는지 먼저 확인"
    exit 1
fi

echo ">> 압축 해제..."
unzip -o "$ZIP"
# 압축 내부 models/smplx/... → <out>/smplx/...
if [ -d models/smplx ]; then rm -rf ./smplx; mv models/smplx ./smplx; rm -rf models; fi

rm -f "$ZIP"
echo ""
echo "✅ 완료:"
ls -la ./smplx/SMPLX_*.npz