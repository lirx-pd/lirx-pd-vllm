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


def build_weight_bindings(model: Any, tapid: Any) -> list[Any]:
    causal_lm = model.language_model if hasattr(model, "language_model") else model
    text_model = causal_lm.model
    bindings = []

    def bind(layer: int | None, role: str, tensor: torch.Tensor) -> None:
        bindings.append(
            tapid.WeightBinding(
                layer=layer,
                role=getattr(tapid.WeightRole, role),
                tensor=tensor,
            )
        )

    def qwen_rms_scale(weight: torch.Tensor) -> torch.Tensor:
        return (weight.float() + 1.0).to(weight.dtype).contiguous()

    for layer_idx, layer in enumerate(text_model.layers):
        if layer.layer_type == "linear_attention":
            mixer = layer.linear_attn
            bind(
                layer_idx,
                "GDN_NORM_SCALE",
                qwen_rms_scale(layer.input_layernorm.weight),
            )
            bind(
                layer_idx,
                "GDN_INPUT_QKVZ",
                mixer.in_proj_qkvz.weight,
            )
            bind(
                layer_idx,
                "GDN_INPUT_BA",
                mixer.in_proj_ba.weight,
            )
            bind(layer_idx, "GDN_CONV", mixer.conv1d.weight[:, 0, :])
            bind(layer_idx, "GDN_A_LOG", mixer.A_log)
            bind(layer_idx, "GDN_DT_BIAS", mixer.dt_bias)
            bind(layer_idx, "GDN_GATE_NORM", mixer.norm.weight)
            bind(layer_idx, "GDN_OUT", mixer.out_proj.weight)
        else:
            assert layer.layer_type == "full_attention"
            mixer = layer.self_attn
            q_gate, key, value = mixer.qkv_proj.weight.split(
                [mixer.q_size * 2, mixer.kv_size, mixer.kv_size], dim=0
            )
            q_gate = q_gate.reshape(
                mixer.num_heads,
                2,
                mixer.head_dim,
                q_gate.shape[1],
            )
            q_gate = torch.cat((q_gate[:, 0], q_gate[:, 1]), dim=0)
            q_gate = q_gate.reshape(mixer.q_size * 2, -1).contiguous()
            bind(
                layer_idx,
                "NORM_SCALE",
                qwen_rms_scale(layer.input_layernorm.weight),
            )
            bind(layer_idx, "ATTN_Q", q_gate)
            bind(layer_idx, "ATTN_K", key)
            bind(layer_idx, "ATTN_V", value)
            bind(layer_idx, "ATTN_O", mixer.o_proj.weight)
            bind(layer_idx, "ATTN_Q_NORM", qwen_rms_scale(mixer.q_norm.weight))
            bind(layer_idx, "ATTN_K_NORM", qwen_rms_scale(mixer.k_norm.weight))

        gate, up = layer.mlp.gate_up_proj.weight.chunk(2, dim=0)
        bind(
            layer_idx,
            "MLP_NORM_SCALE",
            qwen_rms_scale(layer.post_attention_layernorm.weight),
        )
        bind(layer_idx, "MLP_GATE", gate)
        bind(layer_idx, "MLP_UP", up)
        bind(layer_idx, "MLP_DOWN", layer.mlp.down_proj.weight)

    bind(None, "FINAL_NORM", qwen_rms_scale(text_model.norm.weight))
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

    num_computed_cpu = runner.input_batch.num_computed_tokens_cpu[:num_requests]
    num_prompt_cpu = runner.input_batch.num_prompt_tokens[:num_requests]
    if not np.all(num_computed_cpu < num_prompt_cpu):
        raise ValueError("TAPID prefill received a decode request")

    query_start_loc = runner.query_start_loc.gpu[: num_requests + 1]
    num_computed_tokens = runner.num_computed_tokens[:num_requests]
    hidden_output = runner.tapid_hidden_output

    assert hidden_output is not None
    assert hidden_input.dtype == torch.float32 and hidden_input.ndim == 2
    assert hidden_input.shape[1] == runner.model_config.get_hidden_size()
    assert positions.dtype == torch.int64 and positions.shape == (num_tokens,)
    assert query_start_loc.dtype == torch.int32
    assert num_computed_tokens.dtype == torch.int32
    assert state_indices.dtype == torch.int32
    assert hidden_output.shape[0] >= num_tokens

    return tapid.PrefillStep(
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
    )
