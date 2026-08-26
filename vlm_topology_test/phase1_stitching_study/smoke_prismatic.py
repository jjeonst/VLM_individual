"""Smoke test: can we load the Prismatic VLM (already on server) and get an
answer about one HM3D egocentric RGB frame? De-risks 'can we run a VLM here'.

Run:
  HF_HOME=/data/topovlm/caches/huggingface \
  python vlm_topology_test/smoke_prismatic.py
"""
import os, glob
import numpy as np
from PIL import Image

WEIGHTS = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
RGB = "/data/topovlm/habitat/rgb/pr2l_hm3d_objectnav/train/hm3d_v0.2_train_00463-URjpCob8MGw_URjpCob8MGw.basis.glb_24.npy"

import torch
from prismatic import load

hf_token = os.environ.get("HF_TOKEN")
print(f"[load] Prismatic from {WEIGHTS} ...", flush=True)
vlm = load(WEIGHTS, hf_token=hf_token)
vlm.to("cuda", dtype=torch.bfloat16)
vlm.requires_grad_(False)
vlm.eval()
print("[load] done", flush=True)

# one egocentric frame -> PIL
frames = np.load(RGB, mmap_mode="r")
frame = np.asarray(frames[30])  # a mid-episode frame
img = Image.fromarray(frame.astype(np.uint8))
print(f"[img] frame shape {frame.shape}", flush=True)

question = ("You are an agent inside an indoor home, looking straight ahead. "
            "Briefly describe what you see, and say whether there is a doorway or "
            "open passage leading to another room ahead.")
pb = vlm.get_prompt_builder()
pb.add_turn(role="human", message=question)
prompt = pb.get_prompt()

print("[gen] generating ...", flush=True)
with torch.inference_mode():
    answer = vlm.generate(img, prompt, do_sample=False, max_new_tokens=80)
print("\n=== QUESTION ===\n" + question)
print("\n=== PRISMATIC ANSWER ===\n" + str(answer))
print("\n[smoke] OK — VLM loads and generates in this environment.")
