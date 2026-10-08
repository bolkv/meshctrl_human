#!/bin/bash
# bedlam_single 이미지 -> VAE latent(mean/std) precompute, N GPU 병렬.
# 사용: bash precompute_latents_run.sh    (캡션 끝난 뒤 실행)
cd /workspace/shapectrl/data
NGPU=${NGPU:-4}
echo "precompute latents ngpu=$NGPU root=${SINGLE_ROOT:-bedlam_single}"
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i NSHARD=$NGPU SHARD=$i \
    python3 -u precompute_latents_single.py > precompute_latents_shard${i}.log 2>&1 &
  echo "  launched shard $i on GPU $i (PID $!)"
done
echo "진행: tail -f precompute_latents_shard0.log"
wait
echo "ALL SHARDS DONE"
