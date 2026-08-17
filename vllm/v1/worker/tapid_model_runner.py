# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.tapid_qwen3_5 import (
    build_prefill_step,
    build_runtime_bindings,
    build_weight_bindings,
    get_qwen_layer_names,
)


class TapidGPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._validate_config()
        self.tapid_config = vllm_config.additional_config["tapid"]
        self.tapid = importlib.import_module("tapid_vllm")
        self.tapid_session: Any | None = None
        self.tapid_hidden_output: torch.Tensor | None = None
        self.tapid_attention_layers: tuple[str, ...] = ()
        self.tapid_gdn_layers: tuple[str, ...] = ()

    def _validate_config(self) -> None:
        tapid_config = self.vllm_config.additional_config["tapid"]
        if tapid_config.get("model_signature") != "qwen3_5_dense_27b_bf16":
            raise ValueError("TAPID requires the Qwen3.5 27B BF16 signature")
        kv_config = self.vllm_config.kv_transfer_config
        if (
            kv_config is None
            or kv_config.kv_connector != "NixlConnector"
            or kv_config.kv_role != "kv_producer"
        ):
            raise ValueError("TAPID requires a NIXL kv_producer")
        if not self.model_config.enforce_eager:
            raise ValueError("TAPID requires enforce_eager")
        if self.model_config.hf_text_config.model_type != "qwen3_5_text":
            raise ValueError("TAPID requires the dense Qwen3.5 text model")
        if self.model_config.dtype != torch.bfloat16:
            raise ValueError("TAPID requires bfloat16")
        if (
            self.parallel_config.tensor_parallel_size != 1
            or self.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("TAPID P0/P1 requires TP=1 and PP=1")
        if self.speculative_config is not None or self.lora_config is not None:
            raise ValueError("TAPID P0/P1 does not support spec decode or LoRA")
        if self.parallel_config.enable_dbo:
            raise ValueError("TAPID P0/P1 does not support DBO")
        if self.vllm_config.quant_config is not None:
            raise ValueError("TAPID P0/P1 does not support quantization")
        if self.scheduler_config.async_scheduling:
            raise ValueError("TAPID P0/P1 does not support async scheduling")

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
        self.tapid_attention_layers, self.tapid_gdn_layers = get_qwen_layer_names(
            self.vllm_config
        )
        self.tapid_session = self.tapid.Session(
            device=self.device.index,
            model_signature=self.tapid_config["model_signature"],
        )
        self.tapid_session.bind_weights(build_weight_bindings(self.model, self.tapid))
        self.tapid_session.reserve_workspace(
            max_num_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_seqs,
        )
        self.tapid_hidden_output = torch.empty(
            (
                self.scheduler_config.max_num_batched_tokens,
                self.model_config.get_hidden_size(),
            ),
            dtype=self.model_config.dtype,
            device=self.device,
        )

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, is_profiling)
        if not is_profiling:
            assert self.tapid_session is not None
            self.tapid_session.bind_runtime(
                build_runtime_bindings(
                    self,
                    self.tapid,
                    self.tapid_attention_layers,
                    self.tapid_gdn_layers,
                )
            )
            self.tapid_session.start()

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        if self.tapid_session is None or not self.tapid_session.ready:
            raise self.tapid.TapidConfigError(
                "TAPID session is not ready for online forward"
            )
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
        if self.tapid_session is not None:
            self.tapid_session.close()
        self.tapid_hidden_output = None
        super().shutdown()
