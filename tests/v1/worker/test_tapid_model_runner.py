# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import vllm.v1.worker.tapid_model_runner as tapid_runner_module
import vllm.v1.worker.tapid_qwen3_5 as tapid_qwen_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.tapid_model_runner import TapidGPUModelRunner


class TapidConfigError(RuntimeError):
    pass


class WeightRole(Enum):
    NORM_SCALE = auto()
    MLP_NORM_SCALE = auto()
    GDN_NORM_SCALE = auto()
    GDN_INPUT_QKVZ = auto()
    GDN_INPUT_BA = auto()
    GDN_CONV = auto()
    GDN_A_LOG = auto()
    GDN_DT_BIAS = auto()
    GDN_GATE_NORM = auto()
    GDN_OUT = auto()
    ATTN_Q = auto()
    ATTN_K = auto()
    ATTN_V = auto()
    ATTN_O = auto()
    ATTN_Q_NORM = auto()
    ATTN_K_NORM = auto()
    MLP_GATE = auto()
    MLP_UP = auto()
    MLP_DOWN = auto()
    FINAL_NORM = auto()


@dataclass(frozen=True)
class WeightBinding:
    layer: int | None
    role: WeightRole
    tensor: torch.Tensor


@dataclass(frozen=True)
class RuntimeBindings:
    attention_kv: dict[str, torch.Tensor]
    gdn_conv: dict[str, torch.Tensor]
    gdn_recurrent: dict[str, torch.Tensor]
    block_size: int


@dataclass(frozen=True)
class PrefillStep:
    num_tokens: int
    num_requests: int
    hidden_input: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    num_computed_tokens: torch.Tensor
    hidden_output: torch.Tensor
    attention_block_table: torch.Tensor
    attention_slot_mapping: torch.Tensor
    gdn_state_indices: torch.Tensor


class FakeSession:
    def __init__(self, *, device: int | None, model_signature: str):
        self.device = device
        self.model_signature = model_signature
        self.ready = False
        self.weights = None
        self.bindings = None
        self.workspace = None
        self.last_step = None
        self.last_stream = None

    def bind_weights(self, weights):
        self.weights = weights

    def reserve_workspace(self, *, max_num_tokens, max_num_requests):
        self.workspace = (max_num_tokens, max_num_requests)
        return 0

    def bind_runtime(self, bindings):
        self.bindings = bindings

    def start(self):
        self.ready = True

    def run_prefill(self, step, *, stream=None):
        self.last_step = step
        self.last_stream = stream
        return step.hidden_output[: step.num_tokens]

    def close(self):
        self.ready = False
        self.weights = None
        self.bindings = None


FAKE_TAPID = SimpleNamespace(
    Session=FakeSession,
    WeightRole=WeightRole,
    WeightBinding=WeightBinding,
    RuntimeBindings=RuntimeBindings,
    PrefillStep=PrefillStep,
    TapidConfigError=TapidConfigError,
)


def test_tapid_runner_validates_model_signature():
    runner = TapidGPUModelRunner.__new__(TapidGPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        additional_config={"tapid": {"model_signature": "qwen3_5_dense_27b_bf16"}},
        kv_transfer_config=SimpleNamespace(
            kv_connector="NixlConnector",
            kv_role="kv_producer",
        ),
        quant_config=None,
    )
    runner.model_config = SimpleNamespace(
        enforce_eager=True,
        hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
        dtype=torch.bfloat16,
    )
    runner.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_dbo=False,
    )
    runner.speculative_config = None
    runner.lora_config = None
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)

    runner._validate_config()
    runner.vllm_config.additional_config["tapid"]["model_signature"] = "unknown"
    with pytest.raises(ValueError, match="signature"):
        runner._validate_config()


def test_qwen_layer_names_require_27b_layout():
    layers = {
        **{f"model.layers.{index}.self_attn.attn": None for index in range(16)},
        **{f"model.layers.{index}.linear_attn": None for index in range(48)},
    }
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers)
    )

    attention, gdn = tapid_qwen_module.get_qwen_layer_names(config)
    assert len(attention) == 16
    assert len(gdn) == 48

    layers.pop("model.layers.0.linear_attn")
    with pytest.raises(ValueError, match="48 GDN"):
        tapid_qwen_module.get_qwen_layer_names(config)


def test_qwen_weight_bindings_use_loaded_packed_views():
    def linear(weight):
        return SimpleNamespace(weight=weight)

    def norm(size):
        return SimpleNamespace(weight=torch.empty(size))

    def mlp():
        return SimpleNamespace(
            gate_up_proj=linear(torch.arange(24).reshape(6, 4)),
            down_proj=linear(torch.empty(4, 3)),
        )

    gdn_qkvz = torch.empty(8, 4)
    gdn = SimpleNamespace(
        layer_type="linear_attention",
        input_layernorm=norm(4),
        post_attention_layernorm=norm(4),
        mlp=mlp(),
        linear_attn=SimpleNamespace(
            in_proj_qkvz=linear(gdn_qkvz),
            in_proj_ba=linear(torch.empty(2, 4)),
            conv1d=linear(torch.empty(6, 1, 4)),
            A_log=torch.empty(2),
            dt_bias=torch.empty(2),
            norm=norm(2),
            out_proj=linear(torch.empty(4, 3)),
        ),
    )
    packed_qkv = torch.arange(24).reshape(6, 4)
    attention = SimpleNamespace(
        layer_type="full_attention",
        input_layernorm=norm(4),
        post_attention_layernorm=norm(4),
        mlp=mlp(),
        self_attn=SimpleNamespace(
            q_size=2,
            kv_size=1,
            qkv_proj=linear(packed_qkv),
            o_proj=linear(torch.empty(4, 2)),
            q_norm=norm(2),
            k_norm=norm(1),
        ),
    )
    final_norm = norm(4)
    model = SimpleNamespace(
        model=SimpleNamespace(layers=(gdn, attention), norm=final_norm)
    )

    bindings = tapid_qwen_module.build_weight_bindings(model, FAKE_TAPID)
    mapped = {(item.layer, item.role): item.tensor for item in bindings}

    assert len(bindings) == 24
    assert mapped[(0, WeightRole.GDN_INPUT_QKVZ)] is gdn_qkvz
    assert mapped[(0, WeightRole.GDN_CONV)].shape == (6, 4)
    assert mapped[(0, WeightRole.MLP_GATE)].shape == (3, 4)
    assert mapped[(1, WeightRole.ATTN_Q)].data_ptr() == packed_qkv.data_ptr()
    assert mapped[(1, WeightRole.ATTN_K)].shape == (1, 4)
    assert mapped[(1, WeightRole.MLP_UP)].shape == (3, 4)
    assert mapped[(None, WeightRole.FINAL_NORM)] is final_norm.weight


def test_qwen_runtime_and_prefill_bindings(monkeypatch: pytest.MonkeyPatch):
    attention_cache = torch.empty(2, 2)
    conv_state = torch.empty(2, 3)
    recurrent_state = torch.empty(2, 4)
    layers = {
        "model.layers.3.self_attn.attn": SimpleNamespace(kv_cache=attention_cache),
        "model.layers.0.linear_attn": SimpleNamespace(
            kv_cache=(conv_state, recurrent_state)
        ),
    }
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=layers)
        ),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(get_hidden_size=lambda: 5120),
    )
    bindings = tapid_qwen_module.build_runtime_bindings(
        runner,
        FAKE_TAPID,
        ("model.layers.3.self_attn.attn",),
        ("model.layers.0.linear_attn",),
    )
    assert bindings.attention_kv["model.layers.3.self_attn.attn"] is attention_cache
    assert bindings.gdn_conv["model.layers.0.linear_attn"] is conv_state
    assert bindings.gdn_recurrent["model.layers.0.linear_attn"] is recurrent_state

    hidden_input = torch.empty(3, 5120, dtype=torch.bfloat16)
    positions = torch.tensor([5, 6, 7], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    num_computed_tokens = torch.tensor([5, 7], dtype=torch.int32)
    hidden_output = torch.empty(8, 5120, dtype=torch.bfloat16)
    block_table = torch.tensor([[2], [3]], dtype=torch.int32)
    slot_mapping = torch.tensor([33, 34, 48], dtype=torch.int64)
    state_indices = torch.tensor([4, 5], dtype=torch.int32)
    context = SimpleNamespace(
        attn_metadata={
            "attention": SimpleNamespace(
                block_table=block_table,
                slot_mapping=slot_mapping,
            ),
            "gdn": SimpleNamespace(
                num_decodes=1,
                num_spec_decodes=0,
                non_spec_state_indices_tensor=state_indices,
            ),
        }
    )
    monkeypatch.setattr(tapid_qwen_module, "get_forward_context", lambda: context)
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([5, 7], dtype=np.int32),
        num_prompt_tokens=np.array([7, 8], dtype=np.int32),
    )
    runner.query_start_loc = SimpleNamespace(gpu=query_start_loc)
    runner.num_computed_tokens = num_computed_tokens
    runner.tapid_hidden_output = hidden_output

    step = tapid_qwen_module.build_prefill_step(
        runner,
        FAKE_TAPID,
        "attention",
        "gdn",
        hidden_input,
        positions,
    )
    assert step.num_tokens == 3
    assert step.num_requests == 2
    assert step.gdn_state_indices is state_indices
    assert step.attention_block_table is block_table

    runner.input_batch.num_computed_tokens_cpu[1] = 8
    with pytest.raises(ValueError, match="decode request"):
        tapid_qwen_module.build_prefill_step(
            runner,
            FAKE_TAPID,
            "attention",
            "gdn",
            hidden_input,
            positions,
        )


def test_tapid_runner_lifecycle_uses_session():
    runner = TapidGPUModelRunner.__new__(TapidGPUModelRunner)
    runner.tapid = FAKE_TAPID
    runner.tapid_config = {"model_signature": "qwen3_5_dense_27b_bf16"}
    runner.tapid_session = None
    runner.tapid_hidden_output = None
    runner.tapid_attention_layers = ()
    runner.tapid_gdn_layers = ()
    runner.device = torch.device("cpu")
    hidden_input = torch.empty(1, 4, dtype=torch.bfloat16)
    runner.model = SimpleNamespace(embed_input_ids=lambda _: hidden_input)
    runner.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=8,
        max_num_seqs=2,
    )
    runner.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
    )
    runner.vllm_config = SimpleNamespace()
    weight_bindings = [object()]

    with (
        patch.object(GPUModelRunner, "load_model"),
        patch.object(
            tapid_runner_module,
            "get_qwen_layer_names",
            return_value=(("attention",), ("gdn",)),
        ),
        patch.object(
            tapid_runner_module,
            "build_weight_bindings",
            return_value=weight_bindings,
        ),
    ):
        runner.load_model()

    session = runner.tapid_session
    assert session is not None
    assert session.workspace == (8, 2)
    assert session.weights is weight_bindings
    assert runner.tapid_hidden_output.shape == (8, 4)

    bindings = RuntimeBindings({}, {}, {}, 16)
    with (
        patch.object(GPUModelRunner, "initialize_kv_cache"),
        patch.object(
            tapid_runner_module,
            "build_runtime_bindings",
            return_value=bindings,
        ),
    ):
        runner.initialize_kv_cache(SimpleNamespace())
    assert session.ready
    assert session.bindings is bindings

    step = PrefillStep(
        num_tokens=1,
        num_requests=1,
        hidden_input=hidden_input,
        positions=torch.tensor([0], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0], dtype=torch.int32),
        hidden_output=runner.tapid_hidden_output,
        attention_block_table=torch.tensor([[1]], dtype=torch.int32),
        attention_slot_mapping=torch.tensor([0], dtype=torch.int64),
        gdn_state_indices=torch.tensor([1], dtype=torch.int32),
    )
    stream = object()
    with (
        patch.object(
            tapid_runner_module,
            "build_prefill_step",
            return_value=step,
        ) as build_step,
        patch.object(torch.cuda, "current_stream", return_value=stream),
    ):
        output = runner._model_forward(
            torch.tensor([1], dtype=torch.int32), step.positions
        )
    assert output.data_ptr() == runner.tapid_hidden_output.data_ptr()
    assert build_step.call_args.args[-2] is hidden_input
    assert session.last_step is step
    assert session.last_stream is stream

    with patch.object(GPUModelRunner, "shutdown"):
        runner.shutdown()
    assert not session.ready
    assert runner.tapid_hidden_output is None
