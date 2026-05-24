import argparse
import logging
import sys
from concurrent import futures
from pathlib import Path

import grpc
from vllm import LLM, SamplingParams

try:
    proto_dir = Path(__file__).resolve().parents[1] / "vllm_wrapper_proto"
    if str(proto_dir) not in sys.path:
        sys.path.insert(0, str(proto_dir))

    import vllm_wrapper_pb2 as pb2
    import vllm_wrapper_pb2_grpc as pb2_grpc
except ModuleNotFoundError as error:
    raise RuntimeError(
        "Missing generated protobuf modules. Run scripts/gen_vllm_wrapper_proto.sh first."
    ) from error


LOGGER = logging.getLogger("vllm_wrapper_server")
TOP_K = 8


class VllmWrapperService(pb2_grpc.VllmWrapperServicer):
    def __init__(self, llm: LLM):
        self._llm = llm

    def Generate(self, request: pb2.VllmRequest, context: grpc.ServicerContext) -> pb2.VllmResponse:
        try:
            prompt = request.prompt
            if prompt.HasField("token_ids"):
                prompt_token_ids = [int(token_id) for token_id in prompt.token_ids.token_ids]
                prompt_text = None
            elif prompt.HasField("text"):
                prompt_text = prompt.text
                prompt_token_ids = None
            else:
                return pb2.VllmResponse(
                    error=pb2.VllmError(error_message="request.prompt must provide text or token_ids")
                )

            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=list(request.stop),
                include_stop_str_in_output=request.include_stop_str_in_output,
                logprobs=TOP_K if request.requires_logprobs else None,
            )

            outputs = self._llm.generate(
                prompts=[prompt_text] if prompt_text is not None else None,
                prompt_token_ids=[prompt_token_ids] if prompt_token_ids is not None else None,
                sampling_params=sampling_params,
            )

            if len(outputs) != 1 or not outputs[0].outputs:
                return pb2.VllmResponse(
                    error=pb2.VllmError(
                        error_message=f"unexpected vLLM output shape: {len(outputs)}"
                    )
                )

            completion = outputs[0].outputs[0]
            response_text = completion.text

            if not request.requires_logprobs:
                return pb2.VllmResponse(success=pb2.VllmSuccess(response_text=response_text))

            token_ids = [int(token_id) for token_id in completion.token_ids]
            token_logprobs = []
            top_logprobs = []

            completion_logprobs = completion.logprobs or []
            for idx, token_id in enumerate(token_ids):
                if idx < len(completion_logprobs) and completion_logprobs[idx] is not None:
                    token_entry = completion_logprobs[idx]
                    sampled = token_entry.get(token_id)
                    sampled_logprob = (
                        float(getattr(sampled, "logprob", float("-inf")))
                        if sampled is not None
                        else float("-inf")
                    )
                    token_logprobs.append(sampled_logprob)

                    top_candidates = [
                        pb2.VllmLogprob(
                            token_id=int(candidate_token_id),
                            logprob=float(getattr(candidate, "logprob", float("-inf"))),
                        )
                        for candidate_token_id, candidate in token_entry.items()
                    ]
                    top_candidates.sort(key=lambda c: c.logprob, reverse=True)

                    seen_ids = {candidate.token_id for candidate in top_candidates}
                    if token_id not in seen_ids:
                        top_candidates.append(
                            pb2.VllmLogprob(token_id=token_id, logprob=sampled_logprob)
                        )

                    top_candidates = top_candidates[:TOP_K]
                    while len(top_candidates) < TOP_K:
                        top_candidates.append(
                            pb2.VllmLogprob(token_id=token_id, logprob=float("-inf"))
                        )

                    top_logprobs.append(pb2.TopLogprobs(candidates=top_candidates))
                else:
                    token_logprobs.append(float("-inf"))
                    top_logprobs.append(
                        pb2.TopLogprobs(
                            candidates=[
                                pb2.VllmLogprob(token_id=token_id, logprob=float("-inf"))
                                for _ in range(TOP_K)
                            ]
                        )
                    )

            return pb2.VllmResponse(
                success=pb2.VllmSuccess(
                    response_text=response_text,
                    logprobs=pb2.VllmLogprobs(
                        tokens=token_ids,
                        token_logprobs=token_logprobs,
                        top_logprobs=top_logprobs,
                    ),
                )
            )
        except Exception as error:
            LOGGER.exception("Generate failed")
            return pb2.VllmResponse(error=pb2.VllmError(error_message=str(error)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM gRPC wrapper server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--model", required=True, help="vLLM model identifier")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    pb2_grpc.add_VllmWrapperServicer_to_server(VllmWrapperService(llm), grpc_server)

    bind_target = f"{args.host}:{args.port}"
    grpc_server.add_insecure_port(bind_target)
    grpc_server.start()
    LOGGER.info("vLLM wrapper server listening on %s", bind_target)
    grpc_server.wait_for_termination()


if __name__ == "__main__":
    main()
