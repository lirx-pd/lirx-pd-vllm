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
from vllm.v1.worker.tapid_model_runner import (
    TapidGPUModelRunner,
    TapidGPUModelRunnerV2,
)


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
    gdn_state_index_by_layer: torch.Tensor


class FakeSession:
    def __init__(
        self,
        *,
        device: int | None,
        model_signature: str,
        program: str = "full",
        layer: int = 0,
    ):
        self.device = device
        self.model_signature = model_signature
        self.program = program
        self.layer = layer
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
    # Padding stride TAPID requires of hidden_input/hidden_output.
    HIDDEN_ROW_STRIDE=8,
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


def test_qwen_weight_bindings_convert_to_tapid_layout():
    """vLLM keeps Hugging Face ``[out, in]`` linears and zero-centered RMSNorm
    parameters; TAPID's WorkerOps want ``[in, out]`` and a direct scale. Binding
    the loaded views unconverted is a silent wrong-answer bug, so assert the
    conversion rather than tensor identity."""

    def linear(weight):
        return SimpleNamespace(weight=weight.to(torch.bfloat16))

    def norm(size):
        return SimpleNamespace(weight=torch.zeros(size, dtype=torch.bfloat16))

    def mlp():
        return SimpleNamespace(
            gate_up_proj=linear(torch.arange(24.0).reshape(6, 4)),
            down_proj=linear(torch.empty(4, 3)),
        )

    gdn_qkvz = torch.arange(32.0).reshape(8, 4).to(torch.bfloat16)
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
    # Per-head [q, gate] rows, the checkpoint layout TAPID wants split into
    # [all-q, all-gate]: one head of size 2 gives rows q0 q1 g0 g1.
    packed_qkv = torch.arange(24.0).reshape(6, 4).to(torch.bfloat16)
    attention = SimpleNamespace(
        layer_type="full_attention",
        input_layernorm=norm(4),
        post_attention_layernorm=norm(4),
        mlp=mlp(),
        self_attn=SimpleNamespace(
            q_size=2,
            kv_size=1,
            num_heads=1,
            head_dim=2,
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

    # Linears are transposed into K-major, not passed through.
    assert torch.equal(mapped[(0, WeightRole.GDN_INPUT_QKVZ)], gdn_qkvz.t())
    assert mapped[(0, WeightRole.GDN_CONV)].shape == (6, 4)
    assert mapped[(0, WeightRole.MLP_GATE)].shape == (4, 3)
    assert mapped[(1, WeightRole.ATTN_K)].shape == (4, 1)
    assert mapped[(1, WeightRole.MLP_UP)].shape == (4, 3)

    # q_proj is deinterleaved from per-head [q, gate] to [all-q, all-gate].
    q_rows = packed_qkv[:4]
    assert torch.equal(
        mapped[(1, WeightRole.ATTN_Q)],
        torch.cat((q_rows[0:2], q_rows[2:4]), dim=0).t(),
    )

    # Zero-centered RMSNorm parameters become a direct scale of 1 + w.
    assert torch.equal(
        mapped[(None, WeightRole.FINAL_NORM)], torch.ones(4, dtype=torch.bfloat16)
    )
    assert torch.equal(
        mapped[(0, WeightRole.MLP_NORM_SCALE)], torch.ones(4, dtype=torch.bfloat16)
    )
    # ... but the GDN gated norm already stores a direct scale.
    assert torch.equal(
        mapped[(0, WeightRole.GDN_GATE_NORM)], torch.zeros(2, dtype=torch.bfloat16)
    )


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


def init_fake_tapid_state(runner) -> None:
    """Run the real ``_init_tapid_state`` against the fake TAPID module.

    Tests build runners with ``__new__``, so mirroring the initializer keeps
    them in step with the runner's TAPID state instead of restating the
    attribute list at every call site.
    """
    config = SimpleNamespace(
        additional_config={"tapid": {"model_signature": "qwen3_5_dense_27b_bf16"}}
    )
    with patch.object(
        tapid_runner_module.importlib, "import_module", return_value=FAKE_TAPID
    ):
        runner._init_tapid_state(config)


def test_tapid_runner_lifecycle_uses_session():
    runner = TapidGPUModelRunner.__new__(TapidGPUModelRunner)
    init_fake_tapid_state(runner)
    runner.device = torch.device("cpu")
    hidden_input = torch.arange(4.0).reshape(1, 4).to(torch.bfloat16)
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
    assert runner.tapid_runtime_bound
    assert session.bindings is bindings
    # The persistent kernels start lazily at the first TAPID-owned forward:
    # every device-wide sync in vLLM's warmup deadlocks against them.
    assert not session.ready

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
        gdn_state_index_by_layer=torch.full((64,), -1, dtype=torch.int32),
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
    # The embeddings are staged into the padded buffer, not handed over
    # directly: TAPID needs the work slabs' row stride.
    staged = build_step.call_args.args[-2]
    assert staged.data_ptr() == runner.tapid_hidden_input.data_ptr()
    assert staged.stride(0) == FAKE_TAPID.HIDDEN_ROW_STRIDE
    assert torch.equal(staged, hidden_input)
    assert session.last_step is step
    assert session.last_stream is stream

    with patch.object(GPUModelRunner, "shutdown"):
        runner.shutdown()
    assert not session.ready
    assert runner.tapid_hidden_output is None



def test_tapid_v2_prefill_step():
    hidden_input = torch.empty(3, 5120, dtype=torch.bfloat16)
    positions = torch.tensor([5, 6, 7], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    block_table = torch.tensor([[2], [3]], dtype=torch.int32)
    slot_mapping = torch.tensor([33, 34, 48], dtype=torch.int64)
    state_indices = torch.tensor([4, 5], dtype=torch.int32)

    attention_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
    )
    gdn_metadata = SimpleNamespace(
        num_spec_decodes=0,
        num_prefills=1,
        num_decodes=0,
        non_spec_state_indices_tensor=state_indices,
    )
    runner = SimpleNamespace(
        tapid_hidden_output=torch.empty(8, 5120, dtype=torch.bfloat16),
        model_config=SimpleNamespace(get_hidden_size=lambda: 5120),
    )

    step = tapid_qwen_module.build_v2_prefill_step(
        runner, FAKE_TAPID, attention_metadata, gdn_metadata, hidden_input, positions
    )
    assert step.num_tokens == 3
    assert step.num_requests == 2
    assert step.attention_block_table is block_table
    assert step.gdn_state_indices is state_indices

    # Mixed batches are the rejected shape now: a pure-decode step builds the
    # same way a prefill does, and num_computed_tokens is what tells the mixers
    # to continue from the caches rather than start at position 0.
    gdn_metadata.num_decodes = 1
    with pytest.raises(ValueError, match="mixed"):
        tapid_qwen_module.build_v2_prefill_step(
            runner,
            FAKE_TAPID,
            attention_metadata,
            gdn_metadata,
            hidden_input,
            positions,
            torch.full((64,), -1, dtype=torch.int32),
        )
    gdn_metadata.num_prefills = 0
    decode_step = tapid_qwen_module.build_v2_prefill_step(
        runner,
        FAKE_TAPID,
        attention_metadata,
        gdn_metadata,
        hidden_input,
        positions,
        torch.full((64,), -1, dtype=torch.int32),
    )
    assert decode_step.num_tokens == 3


def test_tapid_v2_prefill_step_uses_block_table_tensor_fallback():
    hidden_input = torch.empty(2, 5120, dtype=torch.bfloat16)
    positions = torch.tensor([1, 2], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int32)
    block_table = torch.tensor([[4]], dtype=torch.int32)
    slot_mapping = torch.tensor([9, 10], dtype=torch.int64)
    state_indices = torch.tensor([3], dtype=torch.int32)

    attention_metadata = SimpleNamespace(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )
    gdn_metadata = SimpleNamespace(
        num_spec_decodes=0,
        num_prefills=1,
        num_decodes=0,
        non_spec_state_indices_tensor=state_indices,
    )
    runner = SimpleNamespace(
        tapid_hidden_output=torch.empty(4, 5120, dtype=torch.bfloat16),
        model_config=SimpleNamespace(get_hidden_size=lambda: 5120),
    )

    step = tapid_qwen_module.build_v2_prefill_step(
        runner, FAKE_TAPID, attention_metadata, gdn_metadata, hidden_input, positions
    )
    assert step.attention_block_table is block_table


def test_tapid_v2_runner_lifecycle_uses_session():
    runner = TapidGPUModelRunnerV2.__new__(TapidGPUModelRunnerV2)
    init_fake_tapid_state(runner)
    runner.device = torch.device("cpu")
    hidden_input = torch.arange(4.0).reshape(1, 4).to(torch.bfloat16)
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
        patch.object(tapid_runner_module.GPUModelRunnerV2, "load_model"),
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
    assert runner.tapid_attention_layers == ("attention",)
    assert runner.tapid_gdn_layers == ("gdn",)

    bindings = RuntimeBindings({}, {}, {}, 16)
    with (
        patch.object(tapid_runner_module.GPUModelRunnerV2, "initialize_kv_cache"),
        patch.object(
            tapid_runner_module,
            "build_runtime_bindings",
            return_value=bindings,
        ),
    ):
        runner.initialize_kv_cache(SimpleNamespace())
    assert runner.tapid_runtime_bound
    assert session.bindings is bindings
    # The persistent kernels start lazily at the first TAPID-owned forward:
    # every device-wide sync in vLLM's warmup deadlocks against them.
    assert not session.ready


def test_tapid_v2_runner_forward_routes_prefill_and_decode_and_falls_back():
    runner = TapidGPUModelRunnerV2.__new__(TapidGPUModelRunnerV2)
    init_fake_tapid_state(runner)
    runner.device = torch.device("cpu")
    runner.tapid_attention_layers = ("attention",)
    runner.tapid_gdn_layers = ("gdn",)
    runner.tapid_session = FakeSession(device=None, model_signature="qwen")
    runner.tapid_session.ready = True
    runner.tapid_runtime_bound = True
    runner.tapid_arm()
    hidden_input = torch.arange(4.0).reshape(1, 4).to(torch.bfloat16)
    runner.model = SimpleNamespace(embed_input_ids=lambda _: hidden_input)
    runner.tapid_hidden_output = torch.empty(8, 4, dtype=torch.bfloat16)
    runner.tapid_hidden_input = torch.empty(8, 8, dtype=torch.bfloat16)[:, :4]

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
        gdn_state_index_by_layer=torch.full((64,), -1, dtype=torch.int32),
    )
    stream = object()
    prefill_gdn = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    decode_gdn = SimpleNamespace(
        num_prefills=0,
        num_decodes=1,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    mixed_gdn = SimpleNamespace(
        num_prefills=1,
        num_decodes=1,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    spec_gdn = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    prefill_context = SimpleNamespace(
        attn_metadata={"attention": object(), "gdn": prefill_gdn},
        is_padding=None,
    )

    with (
        patch.object(
            tapid_runner_module,
            "get_forward_context",
            return_value=prefill_context,
        ),
        patch.object(
            tapid_runner_module,
            "build_v2_prefill_step",
            return_value=step,
        ) as build_step,
        patch.object(torch.cuda, "current_stream", return_value=stream),
    ):
        output = runner._model_forward(
            torch.tensor([1], dtype=torch.int32), step.positions
        )
    assert output.data_ptr() == runner.tapid_hidden_output.data_ptr()
    # The embeddings are staged into the padded buffer, not handed over
    # directly: TAPID needs the work slabs' row stride.
    staged = build_step.call_args.args[4]
    assert staged.data_ptr() == runner.tapid_hidden_input.data_ptr()
    assert staged.stride(0) == FAKE_TAPID.HIDDEN_ROW_STRIDE
    assert torch.equal(staged, hidden_input)
    assert runner.tapid_session.last_step is step
    assert runner.tapid_session.last_stream is stream

    # A pure-decode step is TAPID's too: same traversal, one row.
    decode_context = SimpleNamespace(
        attn_metadata={"attention": object(), "gdn": decode_gdn},
        is_padding=None,
    )
    with (
        patch.object(
            tapid_runner_module,
            "get_forward_context",
            return_value=decode_context,
        ),
        patch.object(
            tapid_runner_module, "build_v2_prefill_step", return_value=step
        ),
        patch.object(torch.cuda, "current_stream", return_value=stream),
    ):
        output = runner._model_forward(
            torch.tensor([1], dtype=torch.int32), step.positions
        )
    assert output.data_ptr() == runner.tapid_hidden_output.data_ptr()

    # Mixed and speculative batches still fall back to vLLM's kernels.
    sentinel = object()
    base_calls = []

    def fake_base_forward(self, **kwargs):
        base_calls.append(kwargs)
        return sentinel

    for gdn in (mixed_gdn, spec_gdn):
        fallback_context = SimpleNamespace(
            attn_metadata={"attention": object(), "gdn": gdn},
            is_padding=None,
        )
        with (
            patch.object(
                tapid_runner_module,
                "get_forward_context",
                return_value=fallback_context,
            ),
            patch.object(
                tapid_runner_module.GPUModelRunnerV2,
                "_model_forward",
                new=fake_base_forward,
            ),
        ):
            output = runner._model_forward(
                torch.tensor([1], dtype=torch.int32), step.positions
            )
        assert output is sentinel
    assert len(base_calls) == 2


def test_tapid_preloads_embedding_kernels_before_kernels_go_resident():
    """Lazy CUDA module loading must not be left to happen under the kernels.

    CUDA_MODULE_LOADING defaults to LAZY, so a kernel's module loads on its
    first launch, and TAPID's persistent kernels never exit -- a first launch
    after they go resident blocks in cuLaunchKernel forever. Measured: a
    6-token prompt then a 35-token one hung in F.embedding, while two prompts
    of the same length were fine, because the token count picks the
    specialisation. So the sweep has to cover a spread of counts, and all of it
    has to land before start().
    """
    runner = TapidGPUModelRunnerV2.__new__(TapidGPUModelRunnerV2)
    init_fake_tapid_state(runner)
    runner.device = torch.device("cpu")
    runner.tapid_session = FakeSession(device=None, model_signature="qwen")
    runner.scheduler_config = SimpleNamespace(max_num_batched_tokens=256)

    order: list[str] = []
    embedded: list[int] = []

    def fake_embed(ids):
        order.append("embed")
        embedded.append(int(ids.numel()))
        return torch.zeros(ids.numel(), 4)

    runner.model = SimpleNamespace(embed_input_ids=fake_embed)
    original_start = runner.tapid_session.start

    def tracking_start():
        order.append("start")
        original_start()

    runner.tapid_session.start = tracking_start

    runner._ensure_tapid_started()

    assert order[-1] == "start", order
    assert "embed" in order
    # Both ends of the range, and enough between them to cross whatever
    # threshold the index_select dispatch actually uses.
    assert 1 in embedded and 256 in embedded
    assert len(embedded) >= 4
    assert runner.tapid_session.ready


def test_tapid_v2_runner_never_starts_kernels_before_warmup_completes():
    """The persistent kernels must not go resident during vLLM warmup.

    Warmup and memory profiling are full of device-wide syncs, and once
    TAPID's kernels are running they never return, so any such sync deadlocks
    the worker. Only a fully bound, armed, non-padding, single-request step
    that is purely prefill or purely decode may reach ``start()``.
    """
    runner = TapidGPUModelRunnerV2.__new__(TapidGPUModelRunnerV2)
    init_fake_tapid_state(runner)
    runner.device = torch.device("cpu")
    runner.tapid_attention_layers = ("attention",)
    runner.tapid_gdn_layers = ("gdn",)
    runner.tapid_session = FakeSession(device=None, model_signature="qwen")

    prefill_gdn = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    decode_gdn = SimpleNamespace(
        num_prefills=0,
        num_decodes=1,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    mixed_gdn = SimpleNamespace(
        num_prefills=1,
        num_decodes=1,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    spec_gdn = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )

    def context(is_padding=None, metadata=None):
        return SimpleNamespace(
            attn_metadata=(
                {"attention": object(), "gdn": prefill_gdn}
                if metadata is None
                else metadata
            ),
            is_padding=is_padding,
        )

    multi_req = SimpleNamespace(
        num_prefills=2,
        num_decodes=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([1, 2], dtype=torch.int32),
    )
    cases = {
        # profile_run happens before initialize_kv_cache binds the runtime
        "runtime unbound": (False, False, context()),
        # The GDN write-back keys off request 0 and the scan does not reset at
        # request boundaries, so a multi-request batch must fall back rather
        # than be silently wrong.
        "multi request": (
            True, True,
            context(metadata={"attention": object(), "gdn": multi_req}),
        ),
        # warmup runs real prefill steps, but TAPID is armed only after it
        "not armed": (True, False, context()),
        # dummy runs submit all-padding batches
        "padding batch": (True, True, context(is_padding=torch.tensor([True]))),
        # attention metadata is absent until the KV cache groups exist
        "no metadata": (True, True, context(metadata={})),
        # one traversal cannot carry both shapes, and the write-backs still key
        # off request 0
        "mixed batch": (
            True, True,
            context(metadata={"attention": object(), "gdn": mixed_gdn}),
        ),
        "spec decode": (
            True, True,
            context(metadata={"attention": object(), "gdn": spec_gdn}),
        ),
    }
    for label, (bound, armed, forward_context) in cases.items():
        runner.tapid_runtime_bound = bound
        runner.tapid_armed = armed
        with patch.object(
            tapid_runner_module, "get_forward_context", return_value=forward_context
        ):
            assert not runner._tapid_owns_step(), label
    assert not runner.tapid_session.ready

    runner.tapid_runtime_bound = True
    runner.tapid_arm()
    for label, metadata in (
        ("prefill", None),
        ("decode", {"attention": object(), "gdn": decode_gdn}),
    ):
        with patch.object(
            tapid_runner_module,
            "get_forward_context",
            return_value=context(metadata=metadata),
        ):
            assert runner._tapid_owns_step(), label

    # Verify mode returns vLLM's result after re-running the step, which would
    # advance the GDN recurrence a second time on a decode.
    runner.tapid_verify = True
    with patch.object(
        tapid_runner_module,
        "get_forward_context",
        return_value=context(metadata={"attention": object(), "gdn": decode_gdn}),
    ):
        assert not runner._tapid_owns_step()
    runner.tapid_verify = False
