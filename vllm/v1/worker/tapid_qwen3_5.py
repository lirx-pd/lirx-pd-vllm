# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context


def get_qwen_layer_names(
    vllm_config: VllmConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    layer_names = vllm_config.compilation_config.static_forward_context
    attention = tuple(name for name in layer_names if name.endswith(".self_attn.attn"))
    gdn = tuple(name for name in layer_names if name.endswith(".linear_attn"))
    if len(attention) != 16 or len(gdn) != 48:
        raise ValueError(
            "TAPID expects Qwen3.6-27B with 16 attention and 48 GDN layers; "
            f"got {len(attention)} attention and {len(gdn)} GDN layers"
        )
    return attention, gdn


def _kmajor(weight: torch.Tensor) -> torch.Tensor:
    """Hugging Face Linear ``[out, in]`` -> the TAPID WorkerOp ``[in, out]`` view."""
    assert weight.ndim == 2
    return weight.t().contiguous().to(torch.bfloat16)


def _rms_scale(centered_weight: torch.Tensor) -> torch.Tensor:
    """Qwen3.5 stores zero-centered RMSNorm parameters; TAPID wants the scale.

    vLLM applies these through GemmaRMSNorm / Qwen3NextRMSNorm, both of which
    compute ``x * (1 + w)``, so the ``+1`` lives in the layer rather than the
    checkpoint. TAPID's RmsNorm WorkerOp is plain multiplicative.
    """
    assert centered_weight.ndim == 1
    return (centered_weight.float() + 1.0).to(torch.bfloat16).contiguous()


def _fa_q_gate_kmajor(
    weight: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Deinterleave HF per-head ``[q, gate]`` rows into TAPID ``[all-q, all-gate]``."""
    assert weight.shape[0] == num_heads * head_dim * 2
    heads = weight.reshape(num_heads, 2, head_dim, weight.shape[1])
    q_then_gate = torch.cat((heads[:, 0], heads[:, 1]), dim=0)
    return _kmajor(q_then_gate.reshape(num_heads * head_dim * 2, weight.shape[1]))


def build_weight_bindings(model: Any, tapid: Any) -> list[Any]:
    """Materialize TAPID-layout copies of the vLLM weights and bind them.

    TAPID borrows the device pointers it is given, so every converted tensor
    here stays resident for the process lifetime: budget for a second copy of
    the model on top of vLLM's own weights, which the decode fallback still
    needs. ponytail: a shared layout would remove the copy, but that means
    transposed GEMM WorkerOps on the TAPID side, not a loader change here.
    """
    # Qwen3.5 dense text resolves to Qwen3_5ForCausalLM (model.model) or, for
    # the multimodal conditional-generation entry, to model.language_model.model.
    candidate = getattr(model, "language_model", None) or model
    text_model = getattr(candidate, "model", None) or candidate
    bindings = []

    def bind(layer: int | None, role: str, tensor: torch.Tensor) -> None:
        bindings.append(
            tapid.WeightBinding(
                layer=layer,
                role=getattr(tapid.WeightRole, role),
                tensor=tensor,
            )
        )

    for layer_idx, layer in enumerate(text_model.layers):
        if layer.layer_type == "linear_attention":
            mixer = layer.linear_attn
            # in_proj_qkvz/in_proj_ba are MergedColumnParallelLinear over the
            # checkpoint's separate q/k/v/z and b/a tensors, in that order, so
            # the merged weight transposes straight into TAPID's K-major layout.
            bind(layer_idx, "GDN_NORM_SCALE", _rms_scale(layer.input_layernorm.weight))
            bind(layer_idx, "GDN_INPUT_QKVZ", _kmajor(mixer.in_proj_qkvz.weight))
            bind(layer_idx, "GDN_INPUT_BA", _kmajor(mixer.in_proj_ba.weight))
            bind(
                layer_idx,
                "GDN_CONV",
                mixer.conv1d.weight[:, 0, :].contiguous().to(torch.bfloat16),
            )
            bind(layer_idx, "GDN_A_LOG", mixer.A_log.contiguous().to(torch.bfloat16))
            bind(
                layer_idx,
                "GDN_DT_BIAS",
                mixer.dt_bias.contiguous().to(torch.bfloat16),
            )
            # RMSNormGated stores a direct scale — no zero-centering here.
            bind(
                layer_idx,
                "GDN_GATE_NORM",
                mixer.norm.weight.contiguous().to(torch.bfloat16),
            )
            bind(layer_idx, "GDN_OUT", _kmajor(mixer.out_proj.weight))
        else:
            assert layer.layer_type == "full_attention"
            mixer = layer.self_attn
            q_gate, key, value = mixer.qkv_proj.weight.split(
                [mixer.q_size * 2, mixer.kv_size, mixer.kv_size], dim=0
            )
            bind(layer_idx, "NORM_SCALE", _rms_scale(layer.input_layernorm.weight))
            bind(
                layer_idx,
                "ATTN_Q",
                _fa_q_gate_kmajor(q_gate, mixer.num_heads, mixer.head_dim),
            )
            bind(layer_idx, "ATTN_K", _kmajor(key))
            bind(layer_idx, "ATTN_V", _kmajor(value))
            bind(layer_idx, "ATTN_O", _kmajor(mixer.o_proj.weight))
            bind(layer_idx, "ATTN_Q_NORM", _rms_scale(mixer.q_norm.weight))
            bind(layer_idx, "ATTN_K_NORM", _rms_scale(mixer.k_norm.weight))

        gate, up = layer.mlp.gate_up_proj.weight.chunk(2, dim=0)
        bind(
            layer_idx,
            "MLP_NORM_SCALE",
            _rms_scale(layer.post_attention_layernorm.weight),
        )
        bind(layer_idx, "MLP_GATE", _kmajor(gate))
        bind(layer_idx, "MLP_UP", _kmajor(up))
        bind(layer_idx, "MLP_DOWN", _kmajor(layer.mlp.down_proj.weight))

    bind(None, "FINAL_NORM", _rms_scale(text_model.norm.weight))
    return bindings


def build_runtime_bindings(
    runner: Any,
    tapid: Any,
    attention_layers: tuple[str, ...],
    gdn_layers: tuple[str, ...],
) -> Any:
    layers = runner.vllm_config.compilation_config.static_forward_context
    attention_kv = {name: layers[name].kv_cache for name in attention_layers}
    gdn_states = {name: layers[name].kv_cache for name in gdn_layers}

    for states in gdn_states.values():
        assert len(states) == 2, "GDN cache must contain conv and recurrent states"
        assert states[0].ndim >= 2 and states[1].ndim >= 2

    return tapid.RuntimeBindings(
        attention_kv=attention_kv,
        gdn_conv={name: states[0] for name, states in gdn_states.items()},
        gdn_recurrent={name: states[1] for name, states in gdn_states.items()},
        block_size=runner.cache_config.block_size,
    )


def build_prefill_step(
    runner: Any,
    tapid: Any,
    attention_layer: str,
    gdn_layer: str,
    hidden_input: torch.Tensor,
    positions: torch.Tensor,
) -> Any:
    context = get_forward_context()
    assert isinstance(context.attn_metadata, dict)
    attention_metadata = context.attn_metadata[attention_layer]
    gdn_metadata = context.attn_metadata[gdn_layer]

    if gdn_metadata.num_spec_decodes != 0:
        raise ValueError("TAPID prefill does not support speculative decode")

    state_indices = gdn_metadata.non_spec_state_indices_tensor
    assert state_indices is not None
    num_requests = state_indices.numel()
    num_tokens = hidden_input.shape[0]

    # Qwen3.5 multimodal positions use mrope (3, seq_len); the text row is
    # positions[0].
    if positions.ndim == 2:
        positions = positions[0]

    num_computed_cpu = runner.input_batch.num_computed_tokens_cpu[:num_requests]
    num_prompt_cpu = runner.input_batch.num_prompt_tokens[:num_requests]
    if not np.all(num_computed_cpu < num_prompt_cpu):
        raise ValueError("TAPID prefill received a decode request")

    query_start_loc = runner.query_start_loc.gpu[: num_requests + 1]
    num_computed_tokens = runner.num_computed_tokens[:num_requests]
    hidden_output = runner.tapid_hidden_output

    assert hidden_output is not None
    assert hidden_input.dtype == torch.bfloat16 and hidden_input.ndim == 2
    assert hidden_input.shape[1] == runner.model_config.get_hidden_size()
    assert positions.dtype == torch.int64 and positions.shape == (num_tokens,)
    assert query_start_loc.dtype == torch.int32
    assert num_computed_tokens.dtype == torch.int32
    assert state_indices.dtype == torch.int32
    assert hidden_output.shape[0] >= num_tokens

    step = tapid.PrefillStep(
        num_tokens=num_tokens,
        num_requests=num_requests,
        hidden_input=hidden_input,
        positions=positions,
        query_start_loc=query_start_loc,
        num_computed_tokens=num_computed_tokens,
        hidden_output=hidden_output,
        attention_block_table=attention_metadata.block_table,
        attention_slot_mapping=attention_metadata.slot_mapping,
        gdn_state_indices=state_indices,
        # V1 is not maintained; an all -1 table means "write no GDN state".
        gdn_state_index_by_layer=torch.full(
            (64,), -1, dtype=torch.int32, device=state_indices.device
        ),
    )
    _assert_step_tensors_non_null(
        {
            "hidden_input": hidden_input,
            "hidden_output": hidden_output,
            "positions": positions,
            "query_start_loc": query_start_loc,
            "num_computed_tokens": num_computed_tokens,
            "attention_block_table": attention_metadata.block_table,
            "attention_slot_mapping": attention_metadata.slot_mapping,
            "gdn_state_indices": state_indices,
        }
    )
    return step

def build_gdn_state_index_by_layer(
    gdn_layers: tuple[str, ...],
    attn_metadata: dict[str, Any],
    device: Any,
    max_layers: int = 64,
) -> torch.Tensor:
    """GDN state slot per model layer, -1 where the layer is not GDN.

    vLLM's hybrid allocator hands several GDN layers ONE shared conv/ssm tensor
    and tells them apart by slot index, so the slot is per layer, not per
    request. Passing one request-scoped index makes those layers overwrite each
    other's state and decode then reads whichever layer wrote last.
    """
    table = torch.full((max_layers,), -1, dtype=torch.int32, device="cpu")
    for name in gdn_layers:
        indices = getattr(
            attn_metadata.get(name), "non_spec_state_indices_tensor", None
        )
        if indices is None or indices.numel() == 0 or ".layers." not in name:
            continue
        layer = int(name.split(".layers.")[1].split(".")[0])
        if 0 <= layer < max_layers:
            table[layer] = int(indices[0].item())
    return table.to(device, non_blocking=True)


def build_v2_prefill_step(
    runner: Any,
    tapid: Any,
    attention_metadata: Any,
    gdn_metadata: Any,
    hidden_input: torch.Tensor,
    positions: torch.Tensor,
    gdn_state_index_by_layer: torch.Tensor | None = None,
) -> Any:
    """Build a TAPID prefill step from V2 forward-context metadata."""
    if gdn_metadata.num_spec_decodes != 0:
        raise ValueError("TAPID prefill does not support speculative decode")
    if gdn_metadata.num_decodes != 0:
        raise ValueError("TAPID prefill does not support decode or mixed batches")

    state_indices = gdn_metadata.non_spec_state_indices_tensor
    assert state_indices is not None
    query_start_loc = attention_metadata.query_start_loc
    num_requests = query_start_loc.numel() - 1
    num_tokens = hidden_input.shape[0]

    # Qwen3.5 multimodal positions use mrope (3, seq_len); the text row is
    # positions[0].
    if positions.ndim == 2:
        positions = positions[0]
    # The context length (tokens already in the KV/GDN caches), NOT seq_lens.
    # TAPID keys "continue an existing sequence" off this: handing it the total
    # length makes a fresh prefill resume from whatever the caches happen to
    # hold. Mirrors GDNAttentionMetadata.has_initial_state = context_lens > 0.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_computed_tokens = (
        attention_metadata.seq_lens[:num_requests] - query_lens
    ).to(torch.int32)
    block_table = getattr(attention_metadata, "block_table", None)
    if block_table is None:
        block_table = attention_metadata.block_table_tensor
    slot_mapping = attention_metadata.slot_mapping
    hidden_output = runner.tapid_hidden_output

    assert hidden_output is not None
    assert hidden_input.dtype == torch.bfloat16 and hidden_input.ndim == 2
    assert hidden_input.shape[1] == runner.model_config.get_hidden_size()
    assert positions.dtype == torch.int64 and positions.shape == (num_tokens,)
    assert query_start_loc.dtype == torch.int32
    assert num_computed_tokens.dtype == torch.int32
    assert num_computed_tokens.numel() == num_requests
    assert state_indices.dtype == torch.int32
    assert state_indices.numel() == num_requests
    assert hidden_output.shape[0] >= num_tokens

    return tapid.PrefillStep(
        num_tokens=num_tokens,
        num_requests=num_requests,
        hidden_input=hidden_input,
        positions=positions,
        query_start_loc=query_start_loc,
        num_computed_tokens=num_computed_tokens,
        hidden_output=hidden_output,
        attention_block_table=block_table,
        attention_slot_mapping=slot_mapping,
        gdn_state_indices=state_indices,
        # None means "write no GDN state" — every layer's slot is -1.
        gdn_state_index_by_layer=(
            gdn_state_index_by_layer
            if gdn_state_index_by_layer is not None
            else torch.full((64,), -1, dtype=torch.int32,
                            device=state_indices.device)
        ),
    )


def _assert_step_tensors_non_null(tensors: dict[str, Any]) -> None:
    """Raise with the offending field name instead of a bare null-pointer error."""
    for field, value in tensors.items():
        if value is None:
            raise ValueError(f"TAPID PrefillStep field {field} is None")
        if hasattr(value, "data_ptr") and value.data_ptr() == 0:
            raise ValueError(f"TAPID PrefillStep field {field} has null data_ptr")

