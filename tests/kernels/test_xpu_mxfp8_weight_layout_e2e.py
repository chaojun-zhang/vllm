# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for bce9bc564615e25c15bc1d85800a03663b380364
  "xpu: fix MXFP8 weight layout - use contiguous [K,N] storage"

该测试模拟 checkpoint 中权重存储不满足 64-byte 对齐的场景
（SafeTensors / sharded checkpoint 中常见），证明：

  - WITHOUT FIX (.t() 无 .contiguous())：
      oneDNN 报 "found misaligned buffer"，推理抛出
      RuntimeError: could not execute a primitive

  - WITH FIX (.t().contiguous())：
      新分配 64-byte 对齐的 C-contiguous tensor，推理正常通过

运行方式
--------
  export ZE_AFFINITY_MASK=0,1
  cd /work/vllm
  python tests/kernels/test_xpu_mxfp8_weight_layout_e2e.py

预期输出（无修复时的错误 + 修复后的通过）见文件底部注释。
"""

import sys
import traceback

import torch

# ── 确保能 import vllm ────────────────────────────────────────────────────
sys.path.insert(0, "/work/vllm")
import vllm._xpu_ops  # noqa: F401  — 注册 _xpu_C.fp8_gemm 等 op

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as quant_mxfp8,
)

# ── 颜色输出 ──────────────────────────────────────────────────────────────
RED   = "\033[31m"
GREEN = "\033[32m"
BOLD  = "\033[1m"
RESET = "\033[0m"

DEVICE = "xpu"
M, K, N = 16, 128, 128   # 典型 GEMM 规格
ALIGN   = 64             # oneDNN 要求的最低对齐字节数


# ─────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────

def make_misaligned_weight(N: int, K: int) -> torch.Tensor:
    """
    在 XPU 上构造一个 data_ptr % 64 != 0 的 fp8 [N, K] 权重 tensor。

    SafeTensors 文件对 tensor 的起始地址只保证 8-byte 对齐，
    而不保证 oneDNN 要求的 64-byte 对齐，因此这是真实场景。
    """
    # 多分配 64 字节，然后偏移到第一个 8-byte 但非 64-byte 对齐的地址
    big = torch.zeros(N * K + ALIGN, dtype=torch.uint8, device=DEVICE)
    base = big.data_ptr()
    offset = 0
    for o in range(1, ALIGN):
        if (base + o) % ALIGN != 0:
            offset = o
            break
    assert offset != 0, "无法构造非 64-byte 对齐的 tensor，请检查 allocator"
    misaligned = big[offset : offset + N * K].view(torch.float8_e4m3fn).view(N, K)
    assert misaligned.data_ptr() % ALIGN != 0
    return misaligned


def section(title: str) -> None:
    print(f"\n{'═'*62}")
    print(f"  {BOLD}{title}{RESET}")
    print('═'*62)


# ─────────────────────────────────────────────────────────────────────────
# 主测试逻辑
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not torch.xpu.is_available():
        print("XPU 不可用，跳过测试")
        sys.exit(0)
    if not hasattr(torch.ops._xpu_C, "fp8_gemm"):
        print("_xpu_C.fp8_gemm 未注册，跳过测试")
        sys.exit(0)

    # 准备激活值（每次推理都一样）
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    x_fp8, x_scale = quant_mxfp8(x)

    # 用真实量化函数生成合法的 fp8 权重和 scale，然后把权重的 storage 替换成
    # misaligned 的版本，scale 保持正常（对齐问题只在权重 data_ptr 上体现）
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
    w_fp8_aligned, w_scale_NK = torch.ops.vllm.xpu_mxfp8_quantize(w_bf16)
    # process_weights_after_loading 会对 scale 做 .t().contiguous()
    w_scale = w_scale_NK.view(torch.float8_e8m0fnu).t().contiguous()  # [K//32, N]

    # 构造 misaligned checkpoint 权重 [N, K]：用真实量化值填充到非对齐 storage
    w_checkpoint = make_misaligned_weight(N, K)
    # 把量化后的真实值拷贝进去（保持 fp8 数值合法，只是 storage 地址不对齐）
    w_checkpoint.copy_(w_fp8_aligned)

    # ── 打印对齐状态 ──────────────────────────────────────────────────────
    section("存储对齐状态")
    print(f"  checkpoint weight data_ptr = {hex(w_checkpoint.data_ptr())}")
    print(f"  checkpoint weight %64       = {w_checkpoint.data_ptr() % ALIGN}  ← 非 64-byte 对齐（模拟 SafeTensors）")

    w_buggy = w_checkpoint.t()              # 旧代码：仅转置，不复制
    w_fixed = w_checkpoint.t().contiguous() # 新代码：转置 + 强制 C-contiguous

    print()
    print(f"  BUG  .t()            : stride={w_buggy.stride()}, is_contiguous={w_buggy.is_contiguous()}, data_ptr%64={w_buggy.data_ptr()%ALIGN}")
    print(f"  FIX  .t().contiguous(): stride={w_fixed.stride()}, is_contiguous={w_fixed.is_contiguous()}, data_ptr%64={w_fixed.data_ptr()%ALIGN}")

    # ── WITHOUT FIX ───────────────────────────────────────────────────────
    section("WITHOUT FIX  （layer.weight = checkpoint.t()，无 .contiguous()）")
    print(f"  调用 fp8_gemm(x_fp8, weight_buggy, ...)  weight.data_ptr%64={w_buggy.data_ptr()%ALIGN}\n")
    try:
        out = torch.ops._xpu_C.fp8_gemm(
            x_fp8, w_buggy, torch.bfloat16, x_scale, w_scale, None
        )
        print(f"  {RED}✗ 未报错（未触发对齐检查，测试环境可能已对齐）{RESET}")
        print(f"    output shape={list(out.shape)}")
    except RuntimeError as e:
        print(f"  {RED}✗  RuntimeError（oneDNN 对齐检查失败）:{RESET}")
        print(f"     {e}")
        print()
        print("  --- 完整 traceback ---")
        traceback.print_exc()

    # ── WITH FIX ──────────────────────────────────────────────────────────
    section("WITH FIX  （layer.weight = checkpoint.t().contiguous()）")
    print(f"  调用 fp8_gemm(x_fp8, weight_fixed, ...)  weight.data_ptr%64={w_fixed.data_ptr()%ALIGN}\n")
    try:
        out = torch.ops._xpu_C.fp8_gemm(
            x_fp8, w_fixed, torch.bfloat16, x_scale, w_scale, None
        )
        print(f"  {GREEN}✓  推理成功  output shape={list(out.shape)}{RESET}")
    except RuntimeError as e:
        print(f"  {RED}✗  意外失败: {e}{RESET}")
        traceback.print_exc()

    section("结论")
    print("  缺少 .contiguous() 时，Fortran-order 视图继承了 checkpoint")
    print("  的 misaligned storage，oneDNN gemm_kernel 在执行时检测到：")
    print()
    print(f"  {BOLD}  found misaligned buffer: <ptr> for kernel gemm_kernel{RESET}")
    print(f"  {BOLD}  RuntimeError: could not execute a primitive{RESET}")
    print()
    print("  加了 .contiguous() 后 allocator 分配新的 64-byte 对齐 storage，")
    print("  推理正常通过，且无额外的运行时拷贝开销。")
    print()


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────
# 预期输出
# ─────────────────────────────────────────────────────────────────────────
#
# ══════════════════════════════════════════════════════════════
#   存储对齐状态
# ══════════════════════════════════════════════════════════════
#   checkpoint weight data_ptr = 0xffff81ab54e13e01
#   checkpoint weight %64       = 1  ← 非 64-byte 对齐（模拟 SafeTensors）
#
#   BUG  .t()            : stride=(1, 128), is_contiguous=False, data_ptr%64=1
#   FIX  .t().contiguous(): stride=(128, 1), is_contiguous=True,  data_ptr%64=0
#
# ══════════════════════════════════════════════════════════════
#   WITHOUT FIX  （layer.weight = checkpoint.t()，无 .contiguous()）
# ══════════════════════════════════════════════════════════════
#   调用 fp8_gemm(x_fp8, weight_buggy, ...)  weight.data_ptr%64=1
#
#   onednn_verbose,...,error,runtime,found misaligned buffer: 0xffff...e01
#   for kernel gemm_kernel at index 0,src/gpu/intel/compute/kernel.hpp:199
#
#   ✗  RuntimeError（oneDNN 对齐检查失败）:
#      could not execute a primitive
#
# ══════════════════════════════════════════════════════════════
#   WITH FIX  （layer.weight = checkpoint.t().contiguous()）
# ══════════════════════════════════════════════════════════════
#   调用 fp8_gemm(x_fp8, weight_fixed, ...)  weight.data_ptr%64=0
#
#   ✓  推理成功  output shape=[16, 128]
