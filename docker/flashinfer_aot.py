"""Build-stage helper (Dockerfile, stage "aot"): compile exactly the FlashInfer
kernels OmniVoice's apply_flashinfer path uses, ahead of time, for the GPU
architecture in FLASHINFER_CUDA_ARCH_LIST (e.g. 8.6 = RTX 30), and install them
as FlashInfer AOT artifacts (flashinfer/data/aot).

Why AOT and not a copied JIT cache: FlashInfer only trusts AOT artifacts without
a rebuild; anything under its JIT dir still goes through a ninja/nvcc freshness
check at load time, and the runtime image has no nvcc. The flashinfer-jit-cache
0.7.0.post1+cu130 wheel ships no kernels for this case.

Runs without a GPU: FLASHINFER_CUDA_ARCH_LIST=8.6 sets the target.
Modules (from omnivoice/models/omnivoice_flashinfer.py):
  norm (rmsnorm), rope (apply_rope_pos_ids_inplace), silu_and_mul, and the FA2
  ragged batch prefill planned by PackedAttnRunner: fp16 q/kv/o, int32 indptr,
  head_dim 128, no pos-encoding, no sliding window, no soft cap, no fp16 QK.
"""
import os
import shutil

import torch
import flashinfer.activation as act
import flashinfer.norm as norm
import flashinfer.prefill as prefill
import flashinfer.rope as rope
from flashinfer.jit.core import build_jit_specs

assert os.environ.get("FLASHINFER_CUDA_ARCH_LIST"), "set FLASHINFER_CUDA_ARCH_LIST (e.g. 8.6)"
specs = [
    norm.gen_norm_module(),
    rope.gen_rope_module(),
    act.gen_act_and_mul_module("silu"),
    prefill._gen_batch_prefill_primary_module(
        "fa2", torch.float16, torch.float16, torch.float16, torch.int32, 128, 128, 0, False, False, False),
]
build_jit_specs(specs, verbose=False, skip_prebuilt=False)
for spec in specs:
    dst = spec.aot_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.jit_library_path, dst)
    print(f"AOT {spec.name} -> {dst} ({dst.stat().st_size // 1024} KiB)", flush=True)
