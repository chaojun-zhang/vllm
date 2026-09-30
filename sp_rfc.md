
# [RFC] Eager Sequence Parallelism with Async-TP Fusion

**Author:** [name]
**Status:** Draft
**Date:** [date]

## Motivation

[vllm-project/vllm#43224](https://github.com/vllm-project/vllm/issues/43224) proposes moving fusions currently implemented as `torch.compile` passes (e.g. `AllReduce + RMSNorm[+Quant]`) into model-code-level fused ops, and calls out SP and Async-TP as open questions under this move. Today SP is a compiler pass (`pass_config.enable_sp`) that rewrites `AllReduce + RMSNorm` into `ReduceScatter + RMSNorm + AllGather`.

This RFC proposes an eager-mode SP implementation, together with an optional Async-TP fused backend.

### Goals

- Decouple SP correctness and behavior from the `torch.compile` pipeline.
- Keep SP opt-in and config-driven, with no change to non-SP code paths.
- Make the fused Comm+GEMM pluggable, without touching model code or call sites.

### Non-goals

- Replacing the existing SP pass.
- New collective primitives — reuse what vLLM / `torch.distributed` already provides.
- Any change to quantization semantics.
- Multi-entry or container-shaped boundaries; this RFC covers single column-parallel entry plus single row-parallel exit.

## Overall Design

### Why SP

Under plain TP, every rank holds the full token sequence, and all ops run on full tokens. But only two kinds of ops actually need full tokens:

- **GEMM** — weights are sharded, so each rank needs the full token input to produce its slice of the output.
- **attention** — `softmax(QK^T)V` mixes across positions; a sharded token dim cannot be computed.

Everything else — norm, residual add, activations — is per-token and could be sharded. But under plain TP they run on full tokens on every rank, redundantly.

SP is about moving that element-wise work from full tokens down to `1/TP` tokens.

### SP split and where the gains are

Split the all-reduce into two halves — a reduce-scatter at the region's output boundary and an all-gather at the next region's input boundary — so the element-wise work in between lands on local shards:

```
Non-SP:  row-GEMM -> all-reduce -> residual/norm -> column-GEMM
SP:      row-GEMM -> reduce-scatter -> sharded residual/norm -> all-gather -> column-GEMM
```

SP attaches at the granularity of a *region* — a stretch of computation that enters and leaves on full tokens. SP only adds one collective at each end; the inside of a region is unaware.

The cost of splitting is that each layer goes from one all-reduce to reduce-scatter + all-gather. What is saved is the element-wise work between those two collectives — from full tokens down to `1/TP`.

A standard Transformer layer splits into four segments:

- **`qkv_proj` → RoPE → attention → `o_proj`** — attention's softmax mixes across positions, so a sharded token dim cannot be computed; `qkv_proj` is column-parallel and `o_proj` is row-parallel, so both ends need full tokens. No gain here.
- **residual + RMSNorm after `o_proj`** — sits between the all-reduce of `o_proj` and the `qkv_proj` of the next block, and is element-wise. Gain.
- **`gate_up_proj` → activation → `down_proj`** — `gate_up_proj` is column-parallel and `down_proj` is row-parallel, so the activation in between runs on full tokens. No gain here.
- **residual + RMSNorm after `down_proj`** — sits between the all-reduce of `down_proj` and the `gate_up_proj` of the next block, and is element-wise. Gain.

### Async-TP fusion

The all-gather and reduce-scatter that SP introduces would otherwise sit in front of their adjacent GEMMs, with the communication time right on the critical path. Async-TP overlaps them: while waiting for peers' data, the rank can already start computing on the parts that have arrived.

On the AG side, `fused_all_gather_matmul` fuses the all-gather with the column GEMM. On the RS side, `fused_matmul_reduce_scatter` fuses the row GEMM with the reduce-scatter.

### CUDA graph compatibility

A CUDA graph cannot change branches at replay time, whereas whether SP engages depends on the runtime token count. For the two to be compatible, a given bucket must have a fixed token count.

They are, because before dispatching to a bucket, vLLM pads the token count up to the bucket size — and also up to a multiple of `tp_size` — before `forward` runs. So the SP decision always reads the padded, fixed value: same bucket, same branch, same shapes, capture once and replay safely.

## API Design

### Config

- `enable_sequence_parallel: bool` — master SP switch.
- `use_sequence_parallel: bool` (derived property) — `enable_sequence_parallel and tensor_parallel_size > 1`. `enable_sequence_parallel` alone gates whether boundaries get wrapped at construction time; `use_sequence_parallel` is what `is_sequence_parallel_active` actually checks at runtime, since SP is meaningless at `tp_size == 1`.
- `sequence_parallel_min_tokens: int` — minimum bucket token count for SP to engage; below it, the plain all-reduce path is used.
- `enable_sequence_parallel_fuse_gemm_comms: bool` — enables the Async-TP fused backend at SP boundaries.

### Runtime helpers

- `sp_shard` / `sp_all_gather` / `sp_reduce_scatter` — existing SP helpers, reused directly.
- `sp_padding_mask(is_padding, hidden_states) -> Tensor` — existing helper; shards a padding mask the same way `sp_shard` shards hidden states, for regions (e.g. attention) that need to know which rows in a local chunk are padding rather than real tokens.
- `is_sequence_parallel_active(parallel_config) -> bool` — whether SP should run for the in-flight forward call. Reads the padded cudagraph bucket size, so the same bucket always takes the same branch; returns `False` when there is no live forward context. `parallel_config` must come from a reliable, non-global source (cached on the module at construction), not `get_current_vllm_config()` — the latter is only guaranteed during model construction, not during a real per-request forward call in the serving worker.
- `suspend_sequence_parallel_boundary(suspend_before, resume_after, parallel_config) -> None` — marks `[suspend_before, resume_after]` as a span where SP is suspended and instead runs TP-style, on full tokens. `suspend_before`'s input boundary gathers the incoming local shard back to full tokens before running its own column-parallel GEMM; `resume_after`'s output boundary scatters the row-parallel GEMM's output back down to a local shard. Multiple regions are expressed by multiple calls. Entry and exit must be a single column/row-parallel layer; the names describe the caller-visible role, not the mechanism.
- `suspend_sequence_parallel_module(module, parallel_config) -> None` — a separate function for a region with no linear boundary to hook into: wraps `module.forward` with a plain `sp_all_gather` on entry and `sp_reduce_scatter` on exit, no fused kernel or bias handling.

## Model Integration

Three touch points:

- **Model entry** (first layer only): after embedding, `sp_shard` down to the local chunk.
- **Model forward** (every layer): runs on the local chunk by default, gathering/scattering only around segments that need full tokens. Two ways:
    - *Fine-grained* (preferred): in `__init__`, call `suspend_sequence_parallel_boundary` once per full-token span — e.g. once for qkv/o, once for gate_up/down. `forward` needs no change.
    - *Coarse-grained* (fallback): when entry or exit is not a single column/row-parallel layer (e.g. separate q/k/v), wrap the segment explicitly in `forward` with `sp_all_gather` / `sp_reduce_scatter`, guarded by `is_sequence_parallel_active`. Does not participate in fusion.
- **Model exit** (after the final norm): `sp_all_gather` back to full tokens, guarded by `is_sequence_parallel_active`.

A complete worked example is in Appendix A.

## Implementation Details

### `is_sequence_parallel_active`

```python
def _current_num_tokens() -> int | None:
    if not is_forward_context_available():
        return None

    batch_descriptor = get_forward_context().batch_descriptor
    return batch_descriptor.num_tokens if batch_descriptor else None


def is_sequence_parallel_active(parallel_config: "ParallelConfig") -> bool:
    """Whether sequence parallelism is active for the in-flight forward call.

    ``parallel_config`` must come from a reliable, non-global source (e.g.
    the ``vllm_config`` a model's ``__init__`` receives directly, cached on
    ``self``) -- ``get_current_vllm_config()`` is only guaranteed to be set
    during model construction, not during a real per-request forward call
    in the serving worker process (see ``gpu_worker.py::execute_model``,
    which doesn't wrap itself in ``set_current_vllm_config()``).
    """
    if not parallel_config.use_sequence_parallel:
        return False

    if not is_forward_context_available():
        # No live batch to check against; conservatively do not engage SP.
        return False
    ctx = get_forward_context()
    if ctx.batch_descriptor is None:
        return False

    # Padded cudagraph bucket size, fixed per bucket.
    num_tokens = ctx.batch_descriptor.num_tokens
    return num_tokens >= parallel_config.sequence_parallel_min_tokens
```

### `suspend_sequence_parallel_boundary` / `suspend_sequence_parallel_module`

Two independent functions, not an overload: `suspend_sequence_parallel_boundary`
covers the common column/row-linear boundary pair; `suspend_sequence_parallel_module`
covers a region with no linear boundary to hook into, wrapping the
module's `forward` with a plain `sp_all_gather` on entry and
`sp_reduce_scatter` on exit.

```python
def suspend_sequence_parallel_boundary(
    suspend_before: "ColumnParallelLinear",
    resume_after: "RowParallelLinear",
    parallel_config: "ParallelConfig",
) -> None:
    if not parallel_config.enable_sequence_parallel:
        return

    def _wrap_all_gather(linear: "ColumnParallelLinear") -> None:
        linear_forward = linear.forward
        fused_kernel = None
        if parallel_config.enable_sequence_parallel_fuse_gemm_comms:
            from vllm.model_executor.kernels.fused_comm import (
                init_fused_comm_kernel,
            )
            fused_kernel = init_fused_comm_kernel(linear)

        def forward(x, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
            if not is_sequence_parallel_active(parallel_config):
                # SP off for this call: restore the layer's own gather behavior.
                return linear_forward(x, *args, **kwargs)

            bias = linear.bias if not linear.skip_bias_add else None
            if fused_kernel is not None:
                logger.info_once("SP input boundary: using fused all_gather + gemm")
                output = fused_kernel.apply(x, bias)
            else:
                logger.info_once("SP input boundary: using all_gather + gemm")
                output = linear.quant_method.apply(linear, sp_all_gather(x), bias)

            # Both branches gather (and then GEMM) on the padded, TP-aligned
            # chunk size, which can be a few rows longer than the true token
            # count; trim once here so neither branch has to be trusted to
            # do it on its own.
            full_num_tokens = _current_num_tokens()
            if full_num_tokens is not None:
                output = output[:full_num_tokens]

            if not linear.return_bias:
                return output
            output_bias = linear.bias if linear.skip_bias_add else None
            return output, output_bias

        linear.forward = forward


    def _wrap_reduce_scatter(linear: "RowParallelLinear") -> None:
        linear_forward = linear.forward
        fused_kernel = None
        if parallel_config.enable_sequence_parallel_fuse_gemm_comms:
            from vllm.model_executor.kernels.fused_comm import (
                init_fused_comm_kernel,
            )
            fused_kernel = init_fused_comm_kernel(linear)

        def forward(
                x: torch.Tensor,
        ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
            if not is_sequence_parallel_active(parallel_config):
                return linear_forward(x)

            if linear.input_is_parallel:
                input_parallel = x
            else:
                split_input = split_tensor_along_last_dim(x, num_partitions=linear.tp_size)
                input_parallel = split_input[linear.tp_rank].contiguous()
            bias_ = None if (linear.tp_rank > 0 or linear.skip_bias_add) else linear.bias

            if fused_kernel is not None:
                logger.info_once("SP output boundary: using fused gemm + reduce_scatter")
                output = fused_kernel.apply(input_parallel, bias_)
            else:
                logger.info_once("SP output boundary: using gemm + reduce_scatter")
                output = linear.quant_method.apply(linear, input_parallel, bias_)
                output = sp_reduce_scatter(output)

            if not linear.return_bias:
                return output
            output_bias = linear.bias if linear.skip_bias_add else None
            return output, output_bias

        linear.forward = forward


    _wrap_all_gather(suspend_before)
    _wrap_reduce_scatter(resume_after)


def suspend_sequence_parallel_module(
    module: torch.nn.Module,
    parallel_config: "ParallelConfig",
) -> None:
    if not parallel_config.enable_sequence_parallel:
        return

    module_forward = module.forward

    def forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not is_sequence_parallel_active(parallel_config):
            return module_forward(x, *args, **kwargs)
        output = module_forward(sp_all_gather(x), *args, **kwargs)
        return sp_reduce_scatter(output)

    module.forward = forward
```

### `FusedAllGatherGemmKernel` / `FusedGemmReduceScatterKernel`

Fused kernels are constructed per boundary direction. The two directions share a common `FusedCommLinearKernel` base (binds `layer`, declares `is_supported`, declares the single abstract `apply` op) and specialize it into two subclasses — a `ColumnParallelLinear` entry boundary only ever needs an AG kernel, a `RowParallelLinear` exit boundary only ever needs an RS kernel. A new backend (new platform, new quantization scheme) only needs to implement whichever of the two it supports.

```python
class FusedCommLinearKernel(ABC):
    """Bound to a specific linear layer at construction, the same way
    `layer.quant_method` is."""

    def __init__(self, layer) -> None:
        self.layer = layer

    def is_supported(self) -> tuple[bool, str | None]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, x_shard, bias) -> Tensor:
        """Run the fused comm+GEMM op for this kernel's direction."""


class FusedAllGatherGemmKernel(FusedCommLinearKernel):
    """Bound to a specific ColumnParallelLinear."""

    def __init__(self, layer: ColumnParallelLinear) -> None:
        super().__init__(layer)

    @abstractmethod
    def apply(self, x_shard, bias) -> Tensor:
        """all-gather(x_shard) fused with layer's GEMM;
        wired into suspend_before (entry side)."""


class FusedGemmReduceScatterKernel(FusedCommLinearKernel):
    """Bound to a specific RowParallelLinear."""

    def __init__(self, layer: RowParallelLinear) -> None:
        super().__init__(layer)

    @abstractmethod
    def apply(self, x_shard, bias) -> Tensor:
        """layer's GEMM fused with reduce-scatter of its output;
        wired into resume_after (exit side)."""
```


`init_fused_comm_kernel(linear) -> FusedAllGatherGemmKernel | FusedGemmReduceScatterKernel | None`
dispatches on `linear`'s direction only, delegating each direction's
quant-method dispatch to its own flat helper -- `_init_ag_gemm_kernel`
for `ColumnParallelLinear`, `_init_gemm_rs_kernel` for
`RowParallelLinear` -- instead of one function nesting both the
direction and quant-method `if`/`elif` chains together. Each helper is a
plain `if`/`elif` on `isinstance(linear.quant_method, ...)`, constructing
the matching kernel class directly (`SymmMemAllGatherGemm(linear)`,
etc.). For the FP8 kernels, the quant method's underlying FP8 sub-kernel
(`quant_method.fp8_linear`) is a `FP8ScaledMMLinearKernel` subclass
selected per-platform from a priority list (Cutlass, FlashInfer, Marlin,
ROCm, Aiter/Torch/CPU variants, the XPU-specific
`XPUW8A8FP8LinearKernel`/`XPUW8A16FP8LinearKernel`, ...); the `symm_mem`
FP8 kernels' weight-layout and activation-quantization assumptions only
hold for `XPUW8A8FP8LinearKernel` specifically, so each helper
additionally requires `isinstance(quant_method.fp8_linear,
XPUW8A8FP8LinearKernel)` before constructing `XPUSymmMemAllGatherScaledMM`
/ `XPUSymmMemScaledMMReduceScatter`. A second, CUDA/ROCm-facing FP8
backend (`TorchSymmMemAllGatherScaledMM` /
`TorchSymmMemScaledMMReduceScatter`) is gated the same way, but on
`isinstance(quant_method.fp8_linear, (PerTensorTorchFP8ScaledMMLinearKernel,
RowWiseTorchFP8ScaledMMLinearKernel))`: these are the only
`TorchFP8ScaledMMLinearKernel` subclasses whose scales feed `aten::_scaled_mm`
directly, matching what the fused `symm_mem` op assumes.
`ChannelWiseTorchFP8ScaledMMLinearKernel` is excluded because it instead
dequantizes the GEMM output with the real scales applied afterwards (an
"unfused DQ" workaround for platforms without native rowwise
`_scaled_mm`). That DQ step happens per-rank, before any cross-rank
reduction; replicating it with a fused op (dummy scales + a single
post-hoc multiply) is only valid where the reduction commutes with
scaling, which isn't guaranteed for the reduce-scatter direction under
dynamic (per-rank-dependent) activation quantization, so it's left
unsupported. Any other FP8 sub-kernel falls back to the plain
(non-fused) path.

Both Torch backend kernels also quantize with a dedicated `QuantFP8`
instance that forces `num_token_padding=None`, instead of reusing
`fp8_linear.quant_fp8` directly: `TorchFP8ScaledMMLinearKernel.
get_output_padding` pads the standalone (non-fused) `_scaled_mm` call's
token dim up to a minimum of 17 rows for perf, but the fused `symm_mem`
ops gather/reduce-scatter each rank's quantized shard directly --
padding added independently per rank would land at each rank's shard
boundary instead of the very end, corrupting the result. The padding's
perf motivation doesn't apply here either, since the real GEMM runs on
the gathered (larger) tensor, not the small per-rank shard.

Block-scaled FP8 (`Fp8BlockScaledMMLinearKernel`) is a separate
sub-hierarchy with its own weight/scale layout (`[N, K]` weight, 2D
per-128-block scale tiles). Its `BlockWiseTorchFP8ScaledMMLinearKernel`
subclass -- selected on CUDA/ROCm, and as an XPU fallback when the
oneDNN/Triton block kernels aren't available -- gets a fused backend
(`TorchSymmMemAllGatherBlockScaledMM` /
`TorchSymmMemBlockScaledMMReduceScatter`), gated the same
`isinstance(quant_method.fp8_linear, ...)` way. The two directions are
*not* symmetric, though, because the public `torch.ops.symm_mem.fused_*`
wrappers apply very different scale-shape validation on each side:

- The AllGather side's scale classifier
  (`_check_and_verify_fp8_all_gather_scale_mode`) only recognizes
  per-tensor (`numel() == 1`) and per-row (trailing dim `== 1`) scales,
  and raises `ValueError` for anything else -- including the block
  scale's `[M, ceil(K/128)]` shape whenever `K > 128` (confirmed with a
  real 2-rank run on actual hardware). So `TorchSymmMemAllGatherBlockScaledMM`
  cannot call the public `fused_all_gather_scaled_matmul` at all; it
  instead calls `_pipelined_multi_all_gather_and_consume` directly (the
  private primitive the public op is built on, with no such scale-shape
  restriction) to jointly gather `(x_q, a_scale)` and run one local
  `aten::_scaled_mm.out` per peer shard as it arrives. (An earlier version
  of this kernel called the public op directly; it looked correct under
  mock-based dispatch tests and real-model import checks, but would have
  raised at runtime for any real multi-rank deployment with `K > 128`
  -- a reminder that routing tests alone don't exercise the actual
  collective/scale-shape internals.)
- The ReduceScatter side's check is generic (`A_scale.shape[:-1] ==
  A.shape[:-1]`, no restriction on the trailing dim), so the block
  scale's shape is always valid there; `TorchSymmMemBlockScaledMMReduceScatter`
  keeps calling the public `patched_fused_scaled_matmul_reduce_scatter`
  wrapper directly, same as the per-tensor/rowwise case.

One CUDA-only caveat carries over to the AllGather fix: the standalone
kernel locally pads `M` to a multiple of 4 for `_scaled_mm` inside a
single call; the fused gather has no hook to do that mid-gather, so it
instead requires the *gathered* token count to already be a multiple of
4 (raises instead of miscomputing if not).

`XPUFp8BlockScaledMMKernel` (XPU's block-scaled subclass) and
`XPUMxFp8LinearKernel` (a separate, non-`Fp8ScaledMMLinearKernel`
hierarchy used by `Mxfp8OnlineLinearMethod`) get the same AllGather/
ReduceScatter asymmetry, for the same scale-shape-classifier reason --
not because of any XPU-specific GEMM-op limitation (`torch._scaled_mm`
natively handles block-FP8 and MXFP8 scales on XPU, so no custom oneDNN
plumbing is needed either way):

- AllGather (`XPUSymmMemAllGatherBlockScaledMM` / `XPUSymmMemAllGatherMXFP8`)
  must use `_pipelined_multi_all_gather_and_consume` directly, exactly
  like the CUDA/ROCm block-scaled AllGather kernel above, jointly
  gathering `(x_q, scale)` and running one local `_xpu_fp8_mm_out` (a
  thin wrapper around `torch._scaled_mm`) per peer shard. MXFP8 has one
  extra wrinkle here: its scale tensor's dtype (`float8_e8m0fnu`) isn't
  supported by the collective backend, so it's bitcast to `uint8` for the
  gather and viewed back before the GEMM.
- ReduceScatter (`XPUSymmMemBlockScaledMMReduceScatter` /
  `XPUSymmMemMXFP8ReduceScatter`) has no such restriction and calls the
  public `patched_fused_scaled_matmul_reduce_scatter` wrapper directly,
  same as the CUDA/ROCm case -- no private primitive or custom
  `mm_out_op` needed, since the scale stays local to each rank and is
  never touched by a collective on this side.

`init_fused_comm_kernel`
calls `kernel.is_supported()` on the constructed result and returns
`None` (logging why) if unsupported. Adding a new quant method means
adding a branch to the relevant helper; adding a second backend means
adding a branch to both -- simple and explicit as long as the matrix of
(direction x quant method x backend) stays small; if it grows, a
dict-based dispatch table is the natural next step.

## Performance Expectations

- **Compute side**: norm / residual / activations drop from full tokens to `1/TP`.
- **Comm side**: total data movement is comparable to all-reduce; the gain comes from overlapping communication with the element-wise work.
- **Async-TP fusion**: on supported directions the communication is hidden behind the GEMM; on unsupported directions there is no extra cost.
- **Small batch (decode)**: communication dominates, so SP is skipped via `sequence_parallel_min_tokens`.

## Scope of Implementation

- `vllm/config/parallel.py` / `vllm/engine/arg_utils.py` — three config options.
- `vllm/v1/worker/gpu_model_runner.py` — OR `parallel_config.enable_sequence_parallel` into the existing `pass_config.enable_sp` checks.
- `vllm/models/common/ops/sequence_parallel.py` (existing file) — helpers, `suspend_sequence_parallel_boundary` / `suspend_sequence_parallel_module`.
- `vllm/model_executor/kernels/fused_comm/` (new) — the fused-kernel base classes and `init_fused_comm_kernel` dispatch, with kernel implementations split by quant scheme/backend: `unquantized.py` (`UnquantizedLinearMethod`), `fp8/torch.py` (CUDA/ROCm FP8, via `aten::_scaled_mm`), `fp8/xpu.py` (XPU FP8/MXFP8, via oneDNN `fp8_gemm`).
- `tests/model_executor/kernels/fused_comm/` (new) — 2-rank correctness tests per backend/quant scheme (unquantized, CUDA/ROCm FP8 per-tensor/row-wise/block-scaled, XPU FP8 per-tensor/block-scaled/MXFP8), each comparing the fused kernel's output against the non-fused collective+GEMM baseline.
- Platform-specific variants of the target model — three touch points.

## Open Questions / Limitations

- **LoRA is not accounted for.** `suspend_sequence_parallel_boundary` wraps a linear
  layer by monkeypatching that instance's `.forward`. LoRA instead replaces
  the submodule with a new wrapper object (`replace_submodule` in
  `vllm/lora/model_manager.py`), whose own `forward` reimplements the
  GEMM+collective logic directly rather than calling `base_layer.forward`.
  Once a LoRA adapter attaches to a wrapped layer, the patched `forward` is
  no longer on the call path — SP silently stops suspending at that
  boundary while the rest of the model still assumes it is. Needs either
  an SP-aware LoRA linear wrapper, or a config-time rejection of SP + LoRA
  together until one exists.
- **PP + SP** is out of scope for this RFC; non-last pipeline-parallel
  ranks are expected to assert against enabling SP until pipeline-stage
  boundary sharding/gathering is designed.
- **Fused-backend fallback has no operator-visible signal today** —
  unsupported platform/quant-method combinations fall back silently
  (debug-level log only). Worth a startup-time summary of how many SP
  boundaries actually got a fused kernel.

## Appendix A: Qwen3 Example

```python
class Qwen3DecoderLayer(Qwen3DecoderLayerBase):
    def __init__(self, ...):
        super().__init__(...)
        self.parallel_config = get_current_vllm_config().parallel_config
        if self.parallel_config.use_sequence_parallel:
            suspend_sequence_parallel_boundary(
                suspend_before=self.self_attn.qkv_proj,
                resume_after=self.self_attn.o_proj,
                parallel_config=self.parallel_config,
            )
            suspend_sequence_parallel_boundary(
                suspend_before=self.mlp.gate_up_proj,
                resume_after=self.mlp.down_proj,
                parallel_config=self.parallel_config,
            )

    def forward(self, positions, hidden_states, residual):
        # Model entry: shard once, only on the first layer's call
        # (residual is None only for that call).
        if residual is None and is_sequence_parallel_active(self.parallel_config):
            hidden_states = sp_shard(hidden_states)
        # Model forward: unchanged — qkv_proj/o_proj and
        # gate_up_proj/down_proj already gather/scatter internally.
        return super().forward(positions, hidden_states, residual)


class Qwen3Model(Qwen2Model):
    def forward(self, input_ids, positions, ...):
        result = super().forward(input_ids, positions, ...)
        # Model exit: gather back to full tokens once, after the final norm,
        # right before the result reaches the LM head.
        if not is_sequence_parallel_active(self.parallel_config):
            return result
        full_num_tokens = positions.shape[0]
        return sp_all_gather(result)[:full_num_tokens]
```
