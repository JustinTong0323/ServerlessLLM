# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import gc
import inspect
import logging
import os
import time
import uuid
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import torch
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    EmbeddingRequestOutput,
    PoolingParams,
    PromptType,
    RequestOutput,
    SamplingParams,
)
from vllm.inputs import TokensPrompt
from vllm.utils import Counter

from sllm.serve.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
)

logger = logging.getLogger("ray")


def process_output(output: RequestOutput, model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }
    return api_response


def process_embedding_output(
    outputs: List[EmbeddingRequestOutput], model_name: str
) -> Dict[str, Any]:
    valid_outputs = [output for output in outputs if output is not None]
    query_tokens = sum(len(output.prompt_token_ids) for output in valid_outputs)
    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": output.outputs.embedding,
            }
            for i, output in enumerate(outputs)
        ],
        "model": model_name,
        "usage": {
            "query_tokens": query_tokens,
            "total_tokens": query_tokens,
        },
    }
    return api_response


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            del self.status_dict[request_id]

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


# Note the GPU resource will be decided when the backend is created
class VllmBackend(SllmBackend):
    # This class implements every method in vllm.entrypoints.openai.api_server
    # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
    # except that we use ray.remote instead of @app and we also add a few new methods:
    # - stop: stops every ongoing request and then stops the backend
    # - get_current_tokens: returns a list of all ongoing request tokens
    # - resume_kv_cache: resumes the key-value cache for the given requests
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.request_trace = LLMEngineStatusDict()
        # if trace_debug is True, request trace will not be deleted after completion
        self.trace_debug = backend_config.get("trace_debug", False)

        # Robustly parse booleans from config (accepts True/False or "true"/"false")
        def _to_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "y", "t"}
            return bool(value)

        self.enforce_eager = _to_bool(backend_config.get("enforce_eager", False))
        self.enable_prefix_caching = _to_bool(
            backend_config.get("enable_prefix_caching", True)
        )
        self.task = backend_config.get("task", "auto")

        async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in async_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype
            
        pretrained_model_name_or_path = backend_config.get("pretrained_model_name_or_path")

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            filtered_engine_config["model"] = backend_config.get(
                "pretrained_model_name_or_path"
            )
        else:
            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = os.path.join(storage_path, "vllm", pretrained_model_name_or_path)
            filtered_engine_config["model"] = model_path
            filtered_engine_config["load_format"] = "serverless_llm"

        # NOTE: Automatic enable prefix cachinging
        filtered_engine_config["enforce_eager"] = self.enforce_eager
        filtered_engine_config["enable_prefix_caching"] = (
            self.enable_prefix_caching
        )
        filtered_engine_config["task"] = self.task

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)

        self.engine = None

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            self.status = BackendStatus.RUNNING

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        model_name: str = request_data.pop("model", "vllm-model")
        messages: Dict[Dict[str, str], str] = request_data.pop("messages", [])
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        # If prompt is not provided, construct it from messages
        inputs: Union[str, TokensPrompt] = request_data.pop(
            "prompt", construct_prompt
        )
        if request_data.get("input_tokens") is not None:
            inputs = TokensPrompt(
                prompt_token_ids=request_data.pop("input_tokens"),
            )

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )

        # Extract stream flag before constructing SamplingParams to avoid validation errors
        stream: bool = request_data.pop("stream", False)

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        # Stream results if requested (OpenAI-compatible chunk format)
        if stream:
            previous_text_lengths: Dict[int, int] = {}
            chunks: List[Dict[str, Any]] = []
            
            # Track timing metrics
            request_start_time = time.time()
            first_token_time = None
            token_timestamps = []
            total_tokens_generated = 0

            async for response_output in results_generator:
                await self.request_trace.update_status(request_id, response_output)
                current_time = time.time()
                created_ts = (
                    current_time
                    if response_output.metrics is None
                    else response_output.metrics.arrival_time
                )

                for idx, result in enumerate(response_output.outputs):
                    full_text = result.text or ""
                    prev_len = previous_text_lengths.get(idx, 0)
                    delta_text = full_text[prev_len:]
                    previous_text_lengths[idx] = len(full_text)
                    
                    if delta_text:
                        # Track first token time (TTFT)
                        if first_token_time is None:
                            first_token_time = current_time
                            ttft_ms = (first_token_time - request_start_time) * 1000
                        
                        # Track token generation timestamps for TPOT calculation
                        token_timestamps.append(current_time)
                        total_tokens_generated += len(delta_text.split())  # Approximate token count
                        
                        chunks.append(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": idx,
                                        "delta": {"content": delta_text},
                                        "logprobs": None,
                                        "finish_reason": None,
                                    }
                                ],
                                # Add timing metrics to each chunk
                                "timing": {
                                    "ttft_ms": ttft_ms if first_token_time else None,
                                    "timestamp": current_time,
                                }
                            }
                        )

                if getattr(response_output, "finished", False):
                    # Calculate final metrics
                    final_time = time.time()
                    total_latency_ms = (final_time - request_start_time) * 1000
                    
                    # Calculate TPOT (Time Per Output Token)
                    tpot_ms = None
                    if len(token_timestamps) > 1 and total_tokens_generated > 0:
                        # Calculate average time between tokens
                        inter_token_latencies = [
                            (token_timestamps[i] - token_timestamps[i-1]) * 1000 
                            for i in range(1, len(token_timestamps))
                        ]
                        tpot_ms = sum(inter_token_latencies) / len(inter_token_latencies)
                    
                    for idx, result in enumerate(response_output.outputs):
                        finish_reason = result.finish_reason
                        if finish_reason is not None:
                            chunks.append(
                                {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": idx,
                                            "delta": {},
                                            "logprobs": None,
                                            "finish_reason": finish_reason,
                                        }
                                    ],
                                    # Add final timing metrics
                                    "timing": {
                                        "ttft_ms": ttft_ms if first_token_time else None,
                                        "tpot_ms": tpot_ms,
                                        "total_latency_ms": total_latency_ms,
                                        "total_tokens": total_tokens_generated,
                                        "timestamp": final_time,
                                    }
                                }
                            )

            if not self.trace_debug:
                await self.request_trace.delete_request(request_id)

            return {
                "id": request_id,
                "object": "chat.completion.chunk.list",
                "model": model_name,
                "data": chunks,
            }


        # Non-stream case
        final_output = None
        request_start_time = time.time()
        first_token_time = None
        
        async for response_output in results_generator:
            final_output = response_output
            await self.request_trace.update_status(request_id, response_output)
            
            # Track first token time for non-streaming
            if first_token_time is None and response_output.outputs:
                for result in response_output.outputs:
                    if result.text:  # First non-empty output
                        first_token_time = time.time()
                        break

        assert final_output is not None

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)

        # Calculate timing metrics for non-streaming
        final_time = time.time()
        total_latency_ms = (final_time - request_start_time) * 1000
        ttft_ms = (first_token_time - request_start_time) * 1000 if first_token_time else None
        
        # For non-streaming, TPOT can be estimated from total time and token count
        total_output_tokens = sum(len(result.token_ids) for result in final_output.outputs)
        tpot_ms = None
        if ttft_ms and total_output_tokens > 1:
            decode_time_ms = total_latency_ms - ttft_ms
            tpot_ms = decode_time_ms / (total_output_tokens - 1)

        response = process_output(final_output, model_name)
        
        # Add timing metrics to the response
        response["timing"] = {
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "total_latency_ms": total_latency_ms,
            "total_output_tokens": total_output_tokens,
        }
        
        return response

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        # Abort all requests
        requests = await self.request_trace.return_all_request_ids()
        tasks = [self.engine.abort(request_id) for request_id in requests]
        await asyncio.gather(*tasks)
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        tokens: List[List[int]] = [
            result.prompt_token_ids + result.outputs[0].token_ids
            for result in ongoing_results
        ]
        return tokens

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)

    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if not request_data:
            return {"error": "Request data is missing"}

        request_counter: Counter = Counter()
        pooling_params: PoolingParams = PoolingParams()
        model_name = request_data.get("model", "vllm-model")
        query = request_data.get("input", [])

        if not query:
            return {"error": "No inputs provided"}

        inputs = cast(Union[PromptType, Sequence[PromptType]], query)

        async def process_input(input_data) -> List[EmbeddingRequestOutput]:
            request_id = str(next(request_counter))
            res = self.engine.encode(input_data, pooling_params, request_id)
            return [result async for result in res]

        raw_outputs = await asyncio.gather(
            *[process_input(input_data) for input_data in inputs],
            return_exceptions=True,
        )

        valid_outputs = []
        for output in raw_outputs:
            if isinstance(output, Exception):
                logger.error(f"Error encountered: {output}")
            else:
                valid_outputs.extend(output)

        if not valid_outputs:
            return {"error": "All inputs failed"}

        return process_embedding_output(valid_outputs, model_name)

    async def fine_tuning(self, request_data: Dict[str, Any]):
        raise NotImplementedError(
            "Fine-tuning is not supported in this backend"
        )
