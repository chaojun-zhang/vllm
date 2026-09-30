# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Comm+GEMM kernels for FP8-quantized linear layers, split by
backend: ``xpu`` (oneDNN ``_xpu_C`` ops, falling back to ``torch._scaled_mm``
where supported) and ``cuda`` (CUDA/ROCm, via ``torch._scaled_mm``-only
``aten::_scaled_mm`` symm_mem fused ops)."""
