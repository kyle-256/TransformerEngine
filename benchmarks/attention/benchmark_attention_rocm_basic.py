# This file was modified for portability to AMDGPU
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os, sys, time
import torch

# Ensure the working directory includes TransformerEngine in sys.path
# os.chdir("my_change_code/TransformerEngine/benchmarks/attention")
cwd = os.getcwd()
# print("当前工作目录:", cwd)


if "TransformerEngine" in cwd:
    index = cwd.index("TransformerEngine") + len("TransformerEngine")
    trimmed_path = cwd[:index]
    sys.path.append(trimmed_path)
    print(trimmed_path)
from tests.pytorch.fused_attn.test_fused_attn import (
    ModelConfig,
    _get_attention_backends,
    _run_dot_product_attention,
)


# data type
# dtype = torch.bfloat16
dtype = torch.bfloat16

# checkpointing
ckpt_attn = False
# workspace optimization path for cuDNN attention
workspace_opt = True
# QKV memory layout
qkv_layout = "bshd_bshd_bshd"
# padding between sequences for qkv_format=thd
pad_between_seqs = False
# training mode
is_training = True

model_configs = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,     mask,              bias
    "test_0": ModelConfig(2, 16, 16, 64, 512, 512, 0.0, "no_mask", "no_bias"  ),  # short seq
    "test_1": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "no_bias"),  # longer seq, mask
    # "test_2": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "post_scale_bias"),  # bias
    # "test_3": ModelConfig(2, 32, 4, 128, 8192, 8192, 0.0, "causal", "no_bias"),  # GQA
}


def calculate_snr(signal_tensor, noise_tensor):
    # 计算信号功率
    signal_power = torch.mean(signal_tensor**2)

    # 计算噪声功率（假设噪声为两个张量的差异）
    noise = signal_tensor - noise_tensor
    noise_power = torch.mean(noise**2)

    # 计算信噪比（以分贝为单位）
    snr = 10 * torch.log10(signal_power / noise_power)

    return snr


# Runs benchmark with warmup iterations and profiles using rocprof
def benchmark_dot_product_attention(model, attention, column_name, filename):
    config = model_configs[model]
    # print(config)
    os.environ["USE_BLOCK_FP8_FA"] = "0"
    warmup_iters = 1
    for i in range(warmup_iters):
        attn_fwd_tri, attn_bwd_tri = _run_dot_product_attention(
            dtype,
            config,
            attention,
            ckpt_attn,
            qkv_layout,
            workspace_opt,
            pad_between_seqs,
            is_training,
        )
    for res in attn_bwd_tri:
        print (res.size())
    # print(attn_bwd_tri)
    print("set USE_BLOCK_FP8_FA to 1")
    os.environ["USE_BLOCK_FP8_FA"] = "1"
    warmup_iters = 1
    for i in range(warmup_iters):
        attn_fwd, attn_bwd = _run_dot_product_attention(
            dtype,
            config,
            attention,
            ckpt_attn,
            qkv_layout,
            workspace_opt,
            pad_between_seqs,
            is_training,
        )
    for res in attn_bwd_tri:
        print (res.size())
    print("test profile")
    snr = calculate_snr(attn_fwd_tri, attn_fwd)
    print(snr)
    for i in range(3):
        snr = calculate_snr(attn_bwd[i], attn_bwd_tri[i])
        print(snr)


def main():

    device_id = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_id)
    print(
        f"Device {device_id}: "
        f"{device_properties.name} GPU, "
        f"sm{device_properties.major}{device_properties.minor} compute capability, "
        f"{device_properties.total_memory/1024**3:.1f}GB memory"
    )
    # Benchmarking starts..
    for model in model_configs.keys():
        config = model_configs[model]
        available_backends, fused_attn_backends = _get_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=qkv_layout,
            window_size=config.window_size,
            pad_between_seqs=pad_between_seqs,
        )
        print("available_backends", available_backends)
        flash_attn_supported, fused_attn_supported, unfused_attn_supported = (
            available_backends
        )

        if not (fused_attn_supported or flash_attn_supported):
            print("No attention backend's detected for ", model)
            continue

        print(
            f'Running {model} with {"cuDNN attention" if fused_attn_supported else ""}'
            f'{" and flash-attention" if flash_attn_supported else ""}...'
        )

        (
            filename_flash_attn,
            filename_fused_attn,
            filename_fused_ck,
            filename_fused_aotriton,
        ) = (None, None, None, None)
        # Benchmark for each attention backend

        if fused_attn_supported:
            filename_fused_attn = os.path.join(
                "profiler_outputs/", f"prof_fused_{model}.csv"
            )
            benchmark_dot_product_attention(
                model, "FusedAttention", "FusedAttention Module", filename_fused_attn
            )


if __name__ == "__main__":
    main()
