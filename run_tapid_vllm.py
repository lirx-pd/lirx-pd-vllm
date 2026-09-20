"""End-to-end Qwen3.6-27B inference on vLLM model-runner-v2 + TAPID persistent kernel."""

from __future__ import annotations

import argparse
import os
import sys

MODEL = (
    "/data/models/hub/models--Qwen--Qwen3.6-27B/snapshots/"
    "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--prompt",
        action="append",
        help="Repeatable. Probing several lengths in one process shows how the "
        "per-row pattern moves with the token count.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    # TAPID keeps a second, K-major copy of the weights alongside vLLM's own.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--no-tapid", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run vLLM's forward alongside TAPID's and log the divergence.",
    )
    parser.add_argument(
        "--probe",
        default="full",
        choices=("full", "gdn", "attention", "mlp"),
        help="Run a single-layer TAPID program and compare it to that vLLM layer.",
    )
    parser.add_argument("--probe-layer", type=int, default=0)
    parser.add_argument("--dump-tokens", default="", help="write token ids to JSON")
    parser.add_argument("--state-audit", action="store_true")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    additional_config = (
        {}
        if args.no_tapid
        else {
            "tapid": {
                "model_signature": "qwen3_5_dense_27b_bf16",
                "verify": args.verify,
                "probe": args.probe,
                "probe_layer": args.probe_layer,
                "state_audit": args.state_audit,
            }
        }
    )

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Each probe prompt must produce a full prefill, not a cache hit.
        enable_prefix_caching=args.probe == "full",
        additional_config=additional_config,
    )
    out = llm.generate(
        args.prompt or ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    for o in out:
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)
    if args.dump_tokens:
        import json
        json.dump(
            [
                {"prompt": o.prompt,
                 "token_ids": list(o.outputs[0].token_ids),
                 "text": o.outputs[0].text}
                for o in out
            ],
            open(args.dump_tokens, "w"),
        )
        print(f"tokens -> {args.dump_tokens}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    sys.exit(main())
