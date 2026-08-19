# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import time
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.tapid_qwen3_5 import (
    build_gdn_state_index_by_layer,
    build_prefill_step,
    build_runtime_bindings,
    build_v2_prefill_step,
    build_weight_bindings,
    get_qwen_layer_names,
)

logger = init_logger(__name__)


def validate_tapid_config(runner: Any) -> None:
    tapid_config = runner.vllm_config.additional_config["tapid"]
    if tapid_config.get("model_signature") != "qwen3_5_dense_27b_bf16":
        raise ValueError("TAPID requires the Qwen3.5 27B BF16 signature")
    if not runner.model_config.enforce_eager:
        raise ValueError("TAPID requires enforce_eager")
    if runner.model_config.hf_text_config.model_type != "qwen3_5_text":
        raise ValueError("TAPID requires the dense Qwen3.5 text model")
    if runner.model_config.dtype != torch.bfloat16:
        raise ValueError("TAPID requires bfloat16")
    if (
        runner.parallel_config.tensor_parallel_size != 1
        or runner.parallel_config.pipeline_parallel_size != 1
    ):
        raise ValueError("TAPID P0/P1 requires TP=1 and PP=1")
    if runner.speculative_config is not None or runner.lora_config is not None:
        raise ValueError("TAPID P0/P1 does not support spec decode or LoRA")
    if runner.parallel_config.enable_dbo:
        raise ValueError("TAPID P0/P1 does not support DBO")
    if runner.vllm_config.quant_config is not None:
        raise ValueError("TAPID P0/P1 does not support quantization")


_TORCH_SYNC_ORIGINALS: dict[str, Any] = {}


def _install_stream_only_sync(device: torch.device) -> None:
    """Downgrade device-wide syncs to a sync of the current stream.

    TAPID's persistent kernels never return, so anything that waits for *all*
    work on the device — ``cudaDeviceSynchronize`` under
    ``torch.cuda.synchronize`` / ``torch.accelerator.synchronize`` — blocks
    forever once they are resident. vLLM only ever needs its own submitted work
    to have landed, which a stream sync gives it.

    ponytail: syncs the current stream only. vLLM's side streams (async output
    copy, prefetch) are already joined through events; add an explicit
    per-stream list here if a future call site needs one that is not.
    """
    if _TORCH_SYNC_ORIGINALS:
        return

    def _sync_current_stream(*args: Any, **kwargs: Any) -> None:
        torch.cuda.current_stream(device).synchronize()

    for module in (torch.cuda, torch.accelerator):
        _TORCH_SYNC_ORIGINALS[module.__name__] = module.synchronize
        module.synchronize = _sync_current_stream
    logger.info("TAPID: device-wide CUDA syncs downgraded to current-stream syncs")


def _restore_device_sync() -> None:
    for module in (torch.cuda, torch.accelerator):
        original = _TORCH_SYNC_ORIGINALS.pop(module.__name__, None)
        if original is not None:
            module.synchronize = original


class _TapidModelRunnerBase:
    """Shared TAPID setup for V1 and V2 model runners."""

    tapid: Any
    tapid_config: dict[str, Any]
    tapid_session: Any
    tapid_hidden_output: torch.Tensor | None
    tapid_attention_layers: tuple[str, ...]
    tapid_gdn_layers: tuple[str, ...]

    def _init_tapid_state(self, vllm_config: VllmConfig) -> None:
        self.tapid_config = vllm_config.additional_config["tapid"]
        self.tapid = importlib.import_module("tapid_vllm")
        self.tapid_session = None
        self.tapid_hidden_output = None
        self.tapid_hidden_input = None
        self.tapid_attention_layers = ()
        self.tapid_gdn_layers = ()
        self.tapid_runtime_bound = False
        self.tapid_armed = False
        # TAPID does not write vLLM's KV / GDN caches yet, so decode after a
        # TAPID-owned prefill reads uninitialized state. Verify mode runs both
        # paths, logs the divergence, and returns vLLM's result.
        self.tapid_verify = bool(self.tapid_config.get("verify", False))
        # Probe mode installs a single-layer TAPID program and compares it
        # against that one vLLM layer, attributing a divergence to GDN or to
        # full attention instead of to the folded 64-layer cadence.
        self.tapid_probe = str(self.tapid_config.get("probe", "full"))
        self.tapid_probe_layer = int(self.tapid_config.get("probe_layer", 0))
        self.tapid_state_audit = bool(self.tapid_config.get("state_audit", False))
        self._probe_state_ref = None
        self._probe_state_tensors = None
        self._probe_state_names = None
        self._probe_state_blocks = None
        # Which engine actually ran each forward. TAPID takes prefill; decode
        # stays on vLLM, so a healthy generation shows one TAPID step per
        # request and one vLLM step per generated token.
        self._tapid_steps = 0
        self._vllm_steps = 0

    def _report_tapid_divergence(
        self, tapid_hidden: torch.Tensor, reference: torch.Tensor
    ) -> None:
        # Compared on the host on purpose. Launching further CUDA kernels after
        # a full vLLM forward has run alongside the persistent kernel blocks
        # inside cuLaunchKernel; a plain D2H copy of an already-drained tensor
        # does not.
        a = tapid_hidden.float()
        b = reference[: a.shape[0]].float()
        diff = (a - b).abs()
        cosine = torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        )
        logger.info(
            "TAPID verify: rows=%d max|err|=%.6g mean|err|=%.6g "
            "ref_rms=%.6g tapid_rms=%.6g cosine=%.6f",
            a.shape[0],
            float(diff.max()),
            float(diff.mean()),
            float(b.square().mean().sqrt()),
            float(a.square().mean().sqrt()),
            float(cosine),
        )
        # Row 0 attends only to itself and is the first GDN scan step, so a
        # good row 0 with degrading later rows points at sequence mixing
        # (attention / GDN scan) rather than the per-token path.
        per_row = torch.nn.functional.cosine_similarity(a, b, dim=1)
        logger.info(
            "TAPID verify: per-row cosine=%s",
            " ".join(f"{v:.4f}" for v in per_row.tolist()),
        )
        # The output buffer is zeroed before the run, so an all-zero row means
        # TAPID never wrote it — a row-extent bug, not a numerical one.
        logger.info(
            "TAPID verify: per-row max|tapid|=%s",
            " ".join(f"{v:.4g}" for v in a.abs().amax(dim=1).tolist()),
        )

    def _report_tapid_state_divergence(self) -> None:
        """Compare the GDN state TAPID wrote against vLLM's own.

        Hidden states matching is not enough for decode: decode reads the
        recurrent/conv state, so a prefill that computes the right output but
        leaves the state empty still generates garbage from the second token on.
        """
        if self._probe_state_ref is None or self._probe_state_tensors is None:
            return
        names = self._probe_state_names or ("gdn_conv", "gdn_recurrent")
        tensors = self._probe_state_tensors
        blocks = self._probe_state_blocks
        if blocks is not None:
            kv, blk = blocks
            tensors = (kv[blk],)
        for name, got, want in zip(names, tensors, self._probe_state_ref):
            a = got.float().cpu()
            b = want.float().cpu()
            nz = int((a != 0).sum())
            if nz == 0:
                logger.info("TAPID state: %-14s NOT WRITTEN (all zero)", name)
                continue
            diff = (a - b).abs()
            cos = torch.nn.functional.cosine_similarity(
                a.flatten(), b.flatten(), dim=0
            )
            # Relative error against the local magnitude: max|err| alone cannot
            # separate "one outlier element is wrong" from "a heavy tail of
            # legitimately large values", and those need different fixes.
            scale = b.abs().clamp_min(1e-6)
            rel = (diff / scale)[b.abs() > b.abs().mean()]
            q = torch.tensor([0.5, 0.9, 0.99, 1.0])
            rq = torch.quantile(rel, q) if rel.numel() else torch.zeros(4)
            logger.info(
                "TAPID state: %-14s max|err|=%.6g ref_rms=%.6g tapid_rms=%.6g "
                "cosine=%.6f nonzero=%d/%d rel[p50/p90/p99/max]=%.2e/%.2e/%.2e/%.2e",
                name, float(diff.max()), float(b.square().mean().sqrt()),
                float(a.square().mean().sqrt()), float(cos), nz, a.numel(),
                *[float(v) for v in rq],
            )

    def _audit_tapid_state(self, attention_metadata: Any, gdn_metadata: Any) -> None:
        """Report which layers' caches the full 64-layer program actually wrote.

        The single-layer probes only ever exercise one layer_idx. If the folded
        program mis-keys the runtime cache, every probe still passes while
        decode reads empty state for most layers.
        """
        ctx_layers = self.vllm_config.compilation_config.static_forward_context
        slots = attention_metadata.slot_mapping
        blocks = sorted({int(v) // self.cache_config.block_size
                         for v in slots.cpu().tolist()})
        state_idx = int(gdn_metadata.non_spec_state_indices_tensor[0].item())

        # Integer indexing gives a view, so .cpu() is a plain memcpy. Advanced
        # indexing or a device-side .sum() would launch kernels, and launching
        # those with the persistent kernel resident wedges the worker.
        def host_nonzero(view) -> bool:
            return bool(view.cpu().any())

        empty_kv, empty_conv, empty_rec = [], [], []
        for i, name in enumerate(self.tapid_attention_layers):
            kv = ctx_layers[name].kv_cache
            if isinstance(kv, (list, tuple)):
                kv = kv[0]
            if not any(host_nonzero(kv[b]) for b in blocks):
                empty_kv.append(i)
        for i, name in enumerate(self.tapid_gdn_layers):
            conv, ssm = ctx_layers[name].kv_cache[0], ctx_layers[name].kv_cache[1]
            if not host_nonzero(conv[state_idx]):
                empty_conv.append(i)
            if not host_nonzero(ssm[state_idx]):
                empty_rec.append(i)
        logger.info(
            "TAPID audit: EMPTY kv=%s conv=%s recurrent=%s",
            f"{len(empty_kv)}/{len(self.tapid_attention_layers)}",
            f"{len(empty_conv)}/{len(self.tapid_gdn_layers)}",
            f"{len(empty_rec)}/{len(self.tapid_gdn_layers)}",
        )
        return self._snapshot_state(blocks, state_idx)

    def _snapshot_state(self, blocks, state_idx) -> dict:
        """Host-side copy of every layer's caches (memcpy only, no kernels)."""
        ctx_layers = self.vllm_config.compilation_config.static_forward_context
        snap = {}
        for i, name in enumerate(self.tapid_attention_layers):
            kv = ctx_layers[name].kv_cache
            if isinstance(kv, (list, tuple)):
                kv = kv[0]
            snap[("kv", i)] = torch.stack([kv[b].cpu() for b in blocks])
        for i, name in enumerate(self.tapid_gdn_layers):
            c, m = ctx_layers[name].kv_cache
            snap[("conv", i)] = c[state_idx].cpu()
            snap[("rec", i)] = m[state_idx].cpu()
        return snap

    def _compare_state_snapshots(self, got: dict, want: dict) -> None:
        """Per-layer cache agreement, so a divergence can be pinned to a layer."""
        for kind in ("kv", "conv", "rec"):
            rows = []
            for key in sorted(k for k in got if k[0] == kind):
                a, b = got[key].float(), want[key].float()
                cos = float(torch.nn.functional.cosine_similarity(
                    a.flatten(), b.flatten(), dim=0))
                rows.append((key[1], cos))
            if not rows:
                continue
            bad = [i for i, c in rows if c < 0.99]
            good = [c for _, c in rows if c >= 0.99]
            logger.info(
                "TAPID audit: %-4s layers=%d bad(<0.99)=%d %s | good_min=%.6f",
                kind, len(rows), len(bad), bad,
                min(good) if good else float("nan"),
            )

    def _tapid_text_model(self) -> Any:
        candidate = getattr(self.model, "language_model", None) or self.model
        return getattr(candidate, "model", None) or candidate

    def _capture_tapid_probe_layer(self, model_inputs: dict[str, Any]) -> tuple:
        """Run the vLLM model and capture one decoder layer's true hidden states.

        vLLM fuses the residual add into the next layer's norm, so a layer takes
        and returns ``(hidden, residual)`` and the actual hidden state at a layer
        boundary is their sum. TAPID's probe consumes and produces that sum.

        The capture deliberately happens before the persistent kernels start:
        the tensor copies here are ordinary kernel launches, and launching those
        after a full vLLM forward has run alongside a resident persistent kernel
        blocks in ``cuLaunchKernel``.
        """
        text_model = self._tapid_text_model()
        captured: dict[str, torch.Tensor] = {}

        def hook(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            residual = kwargs.get(
                "residual", args[1] if len(args) > 1 else None
            )
            captured["input"] = hidden if residual is None else hidden + residual
            captured["output"] = output[0] + output[1]

        handle = text_model.layers[self.tapid_probe_layer].register_forward_hook(
            hook, with_kwargs=True
        )
        try:
            reference = super()._model_forward(**model_inputs)
        finally:
            handle.remove()

        # The GDN state caches this layer owns, so the probe can check what TAPID
        # wrote into them against what vLLM's own kernels wrote.
        self._probe_state_ref = None
        self._probe_state_tensors = None
        self._probe_state_names = None
        self._probe_state_blocks = None
        if self.tapid_probe == "attention":
            ctx_layers = self.vllm_config.compilation_config.static_forward_context
            suffix = f".layers.{self.tapid_probe_layer}.self_attn.attn"
            name = next(
                (n for n in self.tapid_attention_layers if n.endswith(suffix)), None
            )
            entry = ctx_layers.get(name) if name else None
            kv = getattr(entry, "kv_cache", None) if entry is not None else None
            if isinstance(kv, (list, tuple)):
                kv = kv[0]
            attn_md = get_forward_context().attn_metadata.get(name)
            slots = getattr(attn_md, "slot_mapping", None)
            if kv is not None and slots is not None and slots.numel() > 0:
                logger.info(
                    "TAPID state: kv_cache shape=%s stride=%s dtype=%s slots=%s",
                    tuple(kv.shape), tuple(kv.stride()), kv.dtype,
                    slots[: min(8, slots.numel())].tolist(),
                )
                # Only the blocks this request touches.
                blk = torch.unique(slots // self.cache_config.block_size)
                self._probe_state_tensors = (kv[blk],)
                self._probe_state_ref = (kv[blk].clone(),)
                kv[blk] = 0
                self._probe_state_names = ("attn_kv",)
                self._probe_state_blocks = (kv, blk)

        if self.tapid_probe == "gdn":
            ctx_layers = self.vllm_config.compilation_config.static_forward_context
            suffix = f".layers.{self.tapid_probe_layer}.linear_attn"
            name = next(
                (n for n in self.tapid_gdn_layers if n.endswith(suffix)), None
            )
            entry = ctx_layers.get(name) if name else None
            gdn_md = get_forward_context().attn_metadata.get(name)
            idx_t = getattr(gdn_md, "non_spec_state_indices_tensor", None)
            if entry is not None and getattr(entry, "kv_cache", None) and (
                idx_t is not None and idx_t.numel() > 0
            ):
                # Only the slot this request uses: the full ssm_state is
                # [num_slots, 48, 128, 128] fp32 (~3 MiB per slot), and copying
                # all of it back is slow enough that shutdown kills the probe.
                slot = int(idx_t[0].item())
                conv = entry.kv_cache[0][slot]
                ssm = entry.kv_cache[1][slot]
                self._probe_state_tensors = (conv, ssm)
                self._probe_state_ref = (conv.clone(), ssm.clone())
                self._probe_state_names = ("gdn_conv", "gdn_recurrent")
                conv.zero_()
                ssm.zero_()

        layer = text_model.layers[self.tapid_probe_layer]
        if self.tapid_probe == "mlp":
            # The control: no mixer, so TAPID runs only the layer's MLP block
            # over the same input the mixer would have seen.
            hidden = captured["input"]
            normed = layer.post_attention_layernorm(hidden)
            if isinstance(normed, tuple):
                normed = normed[0]
            expected_hidden = hidden + layer.mlp(normed)
        else:
            expected_hidden = captured["output"]

        expected = text_model.norm(expected_hidden)
        if isinstance(expected, tuple):
            expected = expected[0]
        # clone(), not contiguous(): the captured tensor can be one of vLLM's
        # reused activation buffers, and TAPID reads it long after the forward
        # that produced it has moved on.
        probe_input = captured["input"].to(torch.bfloat16).clone()
        return reference, probe_input, expected.cpu()

    def tapid_arm(self) -> None:
        """Hand steady-state prefill over to TAPID (called after vLLM warmup)."""
        self.tapid_armed = True

    def _bind_tapid_runtime(self) -> None:
        assert self.tapid_session is not None
        self.tapid_session.bind_runtime(
            build_runtime_bindings(
                self,
                self.tapid,
                self.tapid_attention_layers,
                self.tapid_gdn_layers,
            )
        )
        self.tapid_runtime_bound = True

    def _tapid_owns_step(self) -> bool:
        """True when this forward is a pure-prefill step TAPID can run.

        Everything else — profiling/dummy runs before ``bind_runtime``,
        all-padding batches, decode and mixed batches — stays on the vLLM
        model. The check must run *before* ``_ensure_tapid_started`` so the
        persistent kernels are never launched for a step vLLM will execute.
        """
        if (
            self.tapid_session is None
            or not self.tapid_runtime_bound
            or not self.tapid_armed
        ):
            return False
        context = get_forward_context()
        if not isinstance(context.attn_metadata, dict):
            return False
        gdn_metadata = context.attn_metadata.get(self.tapid_gdn_layers[0])
        attention_metadata = context.attn_metadata.get(self.tapid_attention_layers[0])
        if gdn_metadata is None or attention_metadata is None:
            return False
        if context.is_padding is not None and bool(context.is_padding.any()):
            return False
        # The GDN state write-back keys everything off request 0, and the scan
        # does not reset at request boundaries, so a batch carrying more than
        # one request would be silently wrong rather than merely unsupported.
        indices = gdn_metadata.non_spec_state_indices_tensor
        if indices is None or indices.numel() != 1:
            return False
        # TAPID owns pure-prefill steps only. Decode and mixed batches run on
        # vLLM's own kernels -- TAPID's contribution to them is the KV / GDN
        # state its prefill wrote, not the decode arithmetic itself.
        return gdn_metadata.num_decodes == 0 and gdn_metadata.num_spec_decodes == 0

    def _ensure_tapid_started(self) -> None:
        assert self.tapid_session is not None
        if not self.tapid_session.ready:
            _install_stream_only_sync(self.device)
            self.tapid_session.start()

    def _allocate_tapid_hidden_output(self) -> None:
        hidden_size = self.model_config.get_hidden_size()
        rows = self.scheduler_config.max_num_batched_tokens
        self.tapid_hidden_output = torch.zeros(
            (rows, hidden_size), dtype=self.model_config.dtype, device=self.device
        )
        # The input is the SOP source, and every stage reads its operands at
        # that one row stride, so it must be padded to the work slabs'
        # capacity. The output is only ever written by the terminal at its own
        # stride, so it stays contiguous.
        stride = self.tapid.HIDDEN_ROW_STRIDE
        assert hidden_size <= stride
        self.tapid_hidden_input = torch.zeros(
            (rows, stride), dtype=self.model_config.dtype, device=self.device
        )[:, :hidden_size]

    def _to_tapid_hidden_input(self, hidden: torch.Tensor) -> torch.Tensor:
        """Stage vLLM's contiguous activations into the padded input buffer."""
        assert self.tapid_hidden_input is not None
        staged = self.tapid_hidden_input[: hidden.shape[0]]
        staged.copy_(hidden)
        return staged

    def _close_tapid_session(self) -> None:
        self.tapid_armed = False
        if self.tapid_session is not None:
            self.tapid_session.close()
            self.tapid_session = None
        # Close() stops the persistent kernels, so a real device sync is legal
        # again — and vLLM's own shutdown path needs one.
        _restore_device_sync()
        self.tapid_hidden_output = None
        self.tapid_hidden_input = None


class TapidGPUModelRunner(_TapidModelRunnerBase, GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._validate_config()
        self._init_tapid_state(vllm_config)

    def _validate_config(self) -> None:
        validate_tapid_config(self)

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
        self.tapid_attention_layers, self.tapid_gdn_layers = get_qwen_layer_names(
            self.vllm_config
        )
        self.tapid_session = self.tapid.Session(
            device=self.device.index,
            model_signature=self.tapid_config["model_signature"],
            program=self.tapid_probe,
            layer=self.tapid_probe_layer,
        )
        self.tapid_session.bind_weights(build_weight_bindings(self.model, self.tapid))
        self.tapid_session.reserve_workspace(
            max_num_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_seqs,
        )
        self._allocate_tapid_hidden_output()

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, is_profiling)
        if not is_profiling:
            self._bind_tapid_runtime()

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        if self.tapid_session is None:
            raise self.tapid.TapidConfigError(
                "TAPID session is not ready for online forward"
            )
        # Start the persistent runtime lazily, after vLLM warmup has compiled
        # its Triton kernels (device-wide sync deadlocks against persistent
        # kernels, so they must not be resident during warmup).
        self._ensure_tapid_started()
        if positions is None:
            raise self.tapid.TapidConfigError("TAPID P0/P1 requires positions")
        if inputs_embeds is not None:
            hidden_input = inputs_embeds
        elif input_ids is not None:
            hidden_input = self.model.embed_input_ids(input_ids)
        else:
            raise self.tapid.TapidConfigError(
                "TAPID P0/P1 requires token IDs or input embeddings"
            )

        hidden_input = self._to_tapid_hidden_input(hidden_input)
        step = build_prefill_step(
            self,
            self.tapid,
            self.tapid_attention_layers[0],
            self.tapid_gdn_layers[0],
            hidden_input,
            positions,
        )
        stream = torch.cuda.current_stream(self.device)
        return self.tapid_session.run_prefill(step, stream=stream)

    def shutdown(self) -> None:
        self._close_tapid_session()
        super().shutdown()


class TapidGPUModelRunnerV2(_TapidModelRunnerBase, GPUModelRunnerV2):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._validate_config()
        self._init_tapid_state(vllm_config)

    def _validate_config(self) -> None:
        validate_tapid_config(self)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        super().load_model(load_dummy_weights, *args, **kwargs)
        self.tapid_attention_layers, self.tapid_gdn_layers = get_qwen_layer_names(
            self.vllm_config
        )
        self.tapid_session = self.tapid.Session(
            device=self.device.index,
            model_signature=self.tapid_config["model_signature"],
            program=self.tapid_probe,
            layer=self.tapid_probe_layer,
        )
        self.tapid_session.bind_weights(build_weight_bindings(self.model, self.tapid))
        self.tapid_session.reserve_workspace(
            max_num_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_seqs,
        )
        self._allocate_tapid_hidden_output()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)
        self._bind_tapid_runtime()

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        if not self._tapid_owns_step():
            self._vllm_steps += 1
            return super()._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
        self._tapid_steps += 1
        logger.info(
            "TAPID owns this forward (prefill): tapid_steps=%d vllm_steps=%d",
            self._tapid_steps, self._vllm_steps,
        )

        context = get_forward_context()
        assert isinstance(context.attn_metadata, dict)
        attention_metadata = context.attn_metadata[self.tapid_attention_layers[0]]
        gdn_metadata = context.attn_metadata[self.tapid_gdn_layers[0]]

        model_inputs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
            **model_kwargs,
        }
        probe_expected = None
        if self.tapid_probe != "full":
            reference, hidden_input, probe_expected = (
                self._capture_tapid_probe_layer(model_inputs)
            )

        if positions is None:
            raise self.tapid.TapidConfigError("TAPID P0/P1 requires positions")
        if probe_expected is not None:
            pass
        elif inputs_embeds is not None:
            hidden_input = inputs_embeds
        elif input_ids is not None:
            hidden_input = self.model.embed_input_ids(input_ids)
        else:
            raise self.tapid.TapidConfigError(
                "TAPID P0/P1 requires token IDs or input embeddings"
            )
        # Staged before the runtime starts: this is a plain kernel launch, and
        # they are only reliable while the persistent kernels are not resident.
        hidden_input = self._to_tapid_hidden_input(hidden_input)

        step = build_v2_prefill_step(
            self,
            self.tapid,
            attention_metadata,
            gdn_metadata,
            hidden_input,
            positions,
            build_gdn_state_index_by_layer(
                self.tapid_gdn_layers, context.attn_metadata, self.device
            ),
        )

        # Start the persistent runtime lazily, and only once a step is really
        # TAPID's: the kernels stay resident for the process lifetime and any
        # device-wide sync after that point deadlocks against them. Building the
        # step first keeps its tensor work on the pre-resident side.
        self._ensure_tapid_started()
        stream = torch.cuda.current_stream(self.device)
        if probe_expected is not None:
            # Zeroed so an unwritten output row is distinguishable from a
            # wrongly-computed one.
            self.tapid_hidden_output.zero_()
        tapid_hidden = self.tapid_session.run_prefill(step, stream=stream)

        if probe_expected is not None:
            stream.synchronize()
            logger.info(
                "TAPID probe: program=%s layer=%d",
                self.tapid_probe,
                self.tapid_probe_layer,
            )
            self._report_tapid_divergence(tapid_hidden.cpu(), probe_expected)
            self._report_tapid_state_divergence()
            return reference

        if self.tapid_state_audit:
            stream.synchronize()
            tapid_state = self._audit_tapid_state(attention_metadata, gdn_metadata)
            blocks = sorted({int(v) // self.cache_config.block_size
                             for v in attention_metadata.slot_mapping.cpu().tolist()})
            sidx = int(gdn_metadata.non_spec_state_indices_tensor[0].item())
            # vLLM writes the same slots, so no clearing is needed first.
            super()._model_forward(
                input_ids=input_ids, positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds, **model_kwargs,
            )
            stream.synchronize()
            self._compare_state_snapshots(
                tapid_state, self._snapshot_state(blocks, sidx)
            )

        if not self.tapid_verify:
            return tapid_hidden

        # Verify mode: run the same step on the vLLM model and report the
        # divergence. vLLM's result is what gets returned, because it is also
        # what populates the KV / GDN caches that decode reads — TAPID does not
        # write them yet.
        #
        # Each phase is drained separately: with a persistent kernel resident,
        # a stall could be TAPID's or vLLM's, and one fused sync at the end
        # cannot say which.
        started = time.monotonic()
        stream.synchronize()
        logger.info(
            "TAPID verify: prefill drained in %.1fs", time.monotonic() - started
        )

        tapid_cpu = tapid_hidden.cpu()
        started = time.monotonic()
        reference = super()._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        stream.synchronize()
        logger.info(
            "TAPID verify: vLLM reference drained in %.1fs",
            time.monotonic() - started,
        )
        self._report_tapid_divergence(tapid_cpu, reference.cpu())
        return reference

    def shutdown(self) -> None:
        self._close_tapid_session()
        super().shutdown()
