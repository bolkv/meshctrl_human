import os, sys, glob, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = os.environ.get("QWEN_MODEL", "/workspace/shapectrl/data/models/Qwen2.5-VL-7B-Instruct")
ROOT  = os.environ.get("SINGLE_ROOT", "/workspace/shapectrl/data/bedlam_single")
IMG_BATCH = int(os.environ.get("IMG_BATCH", 6))

Q = ("In 3 detailed sentences, describe that "
     "person: their body pose and current action/movement; their gender, body type, "
     "and appearance; and their clothing. Then note the background environment."
    )


def main():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(MODEL)

    def gen(paths):
        convs = [[{"role": "user", "content": [
                    {"type": "image", "image": "file://" + os.path.abspath(p)},
                    {"type": "text", "text": Q}]}] for p in paths]
        texts = [proc.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
        imgs, vids = process_vision_info(convs)
        inputs = proc(text=texts, images=imgs, videos=vids, padding=True,
                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=180, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return [c.strip() for c in proc.batch_decode(trimmed, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)]

    samples = sorted(glob.glob(f"{ROOT}/*/*/*_b*/img.jpg"))
    tasks = [p for p in samples if not os.path.exists(os.path.join(os.path.dirname(p), "caption.txt"))]
    nshard = int(os.environ.get("NSHARD", "1")); shard = int(os.environ.get("SHARD", "0"))
    tasks = tasks[shard::nshard]
    print(f"[shard {shard}/{nshard}] {len(tasks)} crops to caption", flush=True)

    ndone = nfail = 0
    for i in range(0, len(tasks), IMG_BATCH):
        batch = tasks[i:i + IMG_BATCH]
        try:
            caps = gen(batch)
            for p, cap in zip(batch, caps):
                with open(os.path.join(os.path.dirname(p), "caption.txt"), "w") as f:
                    f.write(cap)
            ndone += len(batch)
        except Exception as e:
            nfail += len(batch); print(f"[FAIL] batch {i}: {e}", file=sys.stderr, flush=True)
        if (i // IMG_BATCH) % 20 == 0:
            print(f"[shard {shard}] {ndone}/{len(tasks)}", flush=True)
    print(f"[shard {shard}] done: {ndone} captioned, {nfail} failed", flush=True)


if __name__ == "__main__":
    main()
