"""
EvalScope integration utilities for ms-swift models.

This module provides a custom ModelAPI implementation that enables batch inference
for evaluation tasks using ms-swift infer engines. It supports both asynchronous
batch processing for throughput and synchronous mode for stricter memory behavior.
"""

import os
import re
import torch
from concurrent.futures import Future
from dataclasses import dataclass
from evalscope.api.messages import ChatMessage as EvalChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.models.utils.openai import chat_choices_from_openai
from queue import Empty, Queue
from threading import Thread
from typing import Any, List, Optional, Tuple

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
from swift.utils import get_logger

logger = get_logger()


@dataclass
class BatchInferInput:
    """
    Container for batch inference input data.

    Holds all necessary information for a single inference request
    that will be processed as part of a batch.
    """
    ms_input: InferRequest  # ms-swift format request
    ms_config: RequestConfig  # ms-swift format configuration
    batch_size: int  # desired batch size for this request
    engine: Any  # inference engine to use


@dataclass
class _QueueItem:
    """
    Internal queue item for batch processing.

    Pairs a batch input with its corresponding future for result delivery.
    """
    input: BatchInferInput
    future: Future[ModelOutput]  # will be resolved with the inference result


# Global variables for batch processing
# These maintain the shared batch processing infrastructure across all model instances
batch_thread: Optional[Thread] = None  # background thread for processing batches
batch_queue: Queue[_QueueItem] = Queue()  # queue of pending inference requests


@register_model_api('swift_custom')
class EvalModel(ModelAPI):
    """
    Custom ModelAPI implementation for ms-swift models with batch inference support.

    This class integrates ms-swift infer engines with EvalScope's evaluation framework,
    providing efficient batch processing for improved evaluation throughput.
    """

    def __init__(
            self,
            model_name: str,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
    ):
        """
        Initialize the EvalModel with ms-swift backend.

        Args:
            model_name: Name of the model for identification
            base_url: Not used in this implementation (for API compatibility)
            api_key: Not used in this implementation (for API compatibility)
            config: Generation configuration with batch settings
            **model_args: Additional arguments including 'model' and 'template'
        """
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        # Extract model-specific arguments from kwargs
        # This pattern allows us to collect known arguments while preserving unknown ones
        def collect_model_arg(name: str) -> Optional[Any]:
            value = model_args.get(name, None)
            if value is not None:
                model_args.pop(name)
            return value

        # Extract required model parameters
        self.model = collect_model_arg('model')  # model path or identifier
        self.template = collect_model_arg('template')  # conversation template
        self.max_batch_size = collect_model_arg('max_batch_size')  # maximum batch size
        self.infer_backend = (collect_model_arg('infer_backend') or 'transformers').lower()
        self.base_model_path = collect_model_arg('base_model_path')
        self.adapters = collect_model_arg('adapters')
        self.checkpoint_root = collect_model_arg('checkpoint_root')
        self.global_step = collect_model_arg('global_step')
        self.eval_sync_mode = self._to_bool(collect_model_arg('eval_sync_mode'), False)

        self.vllm_gpu_memory_utilization = collect_model_arg('vllm_gpu_memory_utilization')
        self.vllm_tensor_parallel_size = collect_model_arg('vllm_tensor_parallel_size')
        self.vllm_pipeline_parallel_size = collect_model_arg('vllm_pipeline_parallel_size')
        self.vllm_max_model_len = collect_model_arg('vllm_max_model_len')
        self.vllm_max_num_seqs = collect_model_arg('vllm_max_num_seqs')
        self.vllm_reserved_memory_gb = collect_model_arg('vllm_reserved_memory_gb')
        self.vllm_enforce_eager = collect_model_arg('vllm_enforce_eager')
        self.vllm_enable_prefix_caching = collect_model_arg('vllm_enable_prefix_caching')
        self.vllm_disable_custom_all_reduce = collect_model_arg('vllm_disable_custom_all_reduce')

        # Initialize the inference engine with batch support.
        # Keep Transformers as default behavior for backward compatibility.
        if self.infer_backend == 'vllm':
            self.engine = self._build_vllm_engine_or_fallback()
        else:
            self.engine = TransformersEngine(self.model, template=self.template, max_batch_size=self.max_batch_size)

    @staticmethod
    def _extract_checkpoint_step(path: str) -> int:
        match = re.search(r'checkpoint-(\d+)$', path)
        return int(match.group(1)) if match else -1

    def _list_adapter_checkpoints(self) -> List[str]:
        if not self.checkpoint_root or not os.path.isdir(self.checkpoint_root):
            return []
        ckpts = []
        for item in os.scandir(self.checkpoint_root):
            if not item.is_dir() or not item.name.startswith('checkpoint-'):
                continue
            # LoRA checkpoint marker file
            if os.path.isfile(os.path.join(item.path, 'adapter_config.json')):
                ckpts.append(item.path)
        return sorted(ckpts, key=self._extract_checkpoint_step)

    def _resolve_adapter_path(self) -> Optional[str]:
        if isinstance(self.adapters, str) and self.adapters:
            logger.info(f'Using explicit adapter path from args: {self.adapters}')
            return self.adapters
        if isinstance(self.adapters, list) and self.adapters:
            logger.info(f'Using explicit adapter path from args: {self.adapters[0]}')
            return self.adapters[0]

        current_step = self._to_int(self.global_step, -1)
        if self.checkpoint_root and current_step >= 0:
            current_ckpt = os.path.join(self.checkpoint_root, f'checkpoint-{current_step}')
            if os.path.isfile(os.path.join(current_ckpt, 'adapter_config.json')):
                logger.info(f'Using current-step adapter checkpoint: {current_ckpt}')
                return current_ckpt
            logger.warning(
                f'Current-step checkpoint checkpoint-{current_step} not found under {self.checkpoint_root}. '
                'Skip vLLM LoRA loading for this eval step to avoid using stale checkpoints.')
            return None

        ckpts = self._list_adapter_checkpoints()
        if not ckpts:
            logger.warning('No LoRA checkpoint found under checkpoint_root for EvalModel.')
            return None

        selected_ckpt = ckpts[-1]
        logger.info(f'Using latest adapter checkpoint: {selected_ckpt}')
        return selected_ckpt

    def _resolve_model_path(self) -> Optional[str]:
        if isinstance(self.model, str):
            return self.model
        if self.base_model_path:
            return self.base_model_path
        model_dir = getattr(self.model, 'model_dir', None)
        if model_dir:
            return model_dir
        return None

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _to_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    @staticmethod
    def _is_oom_exception(ex: Exception) -> bool:
        if isinstance(ex, torch.cuda.OutOfMemoryError):
            return True
        msg = str(ex).lower()
        return 'out of memory' in msg or 'cuda error: out of memory' in msg

    def _choose_safe_vllm_utilization(self) -> float:
        requested = self._to_float(self.vllm_gpu_memory_utilization, 0.9)
        requested = min(max(requested, 0.1), 0.95)
        if not torch.cuda.is_available():
            return requested

        torch.cuda.empty_cache()
        device_id = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)

        reserved_gb = self._to_float(self.vllm_reserved_memory_gb, 2.0)
        reserved_bytes = int(max(reserved_gb, 0.) * 1024**3)
        safe_bytes = max(free_bytes - reserved_bytes, 0)

        if total_bytes <= 0:
            return requested

        safe_util = safe_bytes / total_bytes
        safe_util = min(requested, safe_util)
        safe_util = min(max(safe_util, 0.1), 0.95)

        logger.info(
            f'vLLM memory auto-tune: free={free_bytes / 1024**3:.2f}GB, '
            f'total={total_bytes / 1024**3:.2f}GB, reserved={reserved_gb:.2f}GB, '
            f'safe_max={safe_bytes / 1024**3:.2f}GB, gpu_memory_utilization={safe_util:.3f}')
        return safe_util

    def _build_vllm_engine_or_fallback(self):
        model_path = self._resolve_model_path()
        adapter_path = self._resolve_adapter_path()

        if not model_path:
            logger.warning('infer_backend=vllm but no model path found; fallback to TransformersEngine.')
            return TransformersEngine(self.model, template=self.template, max_batch_size=self.max_batch_size)
        if not adapter_path:
            logger.warning('infer_backend=vllm but no LoRA checkpoint found; fallback to TransformersEngine.')
            return TransformersEngine(self.model, template=self.template, max_batch_size=self.max_batch_size)

        from swift.infer_engine import VllmEngine

        tensor_parallel_size = self._to_int(self.vllm_tensor_parallel_size, 1)
        pipeline_parallel_size = self._to_int(self.vllm_pipeline_parallel_size, 1)
        max_num_seqs = self._to_int(self.vllm_max_num_seqs, 1)
        max_model_len = None if self.vllm_max_model_len is None else self._to_int(self.vllm_max_model_len, 0)
        if max_model_len == 0:
            max_model_len = None

        base_gpu_memory_utilization = self._choose_safe_vllm_utilization()
        enforce_eager = self._to_bool(self.vllm_enforce_eager, False)
        enable_prefix_caching = self.vllm_enable_prefix_caching
        disable_custom_all_reduce = self._to_bool(self.vllm_disable_custom_all_reduce, True)

        # GRPO-style fallback strategy: reduce concurrency and memory pressure progressively.
        attempts = [(base_gpu_memory_utilization, max_num_seqs, max_model_len)]
        if max_num_seqs > 1:
            attempts.append((min(base_gpu_memory_utilization, 0.7), 1, max_model_len))
        if max_model_len is None or max_model_len > 4096:
            attempts.append((min(base_gpu_memory_utilization, 0.6), 1, 4096))
        attempts.append((min(base_gpu_memory_utilization, 0.45), 1, 3072))

        for attempt_idx, (gpu_memory_utilization, attempt_max_num_seqs, attempt_max_model_len) in enumerate(attempts, 1):
            logger.info(
                f'Using VllmEngine for eval (attempt {attempt_idx}/{len(attempts)}). '
                f'model={model_path}, adapter={adapter_path}, '
                f'gpu_memory_utilization={gpu_memory_utilization}, '
                f'max_num_seqs={attempt_max_num_seqs}, max_model_len={attempt_max_model_len}')
            try:
                return VllmEngine(
                    model_path,
                    template=self.template,
                    adapters=[adapter_path],
                    use_async_engine=False,
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    max_model_len=attempt_max_model_len,
                    max_num_seqs=attempt_max_num_seqs,
                    enforce_eager=enforce_eager,
                    enable_prefix_caching=enable_prefix_caching,
                    disable_custom_all_reduce=disable_custom_all_reduce,
                )
            except Exception as ex:
                if self._is_oom_exception(ex):
                    logger.warning(
                        f'vLLM init OOM on attempt {attempt_idx}/{len(attempts)}: {ex}. '
                        'Retry with stricter memory settings.')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        logger.warning('All vLLM init attempts failed due to OOM; fallback to TransformersEngine for eval.')
        return TransformersEngine(self.model, template=self.template, max_batch_size=self.max_batch_size)

    def generate(
        self,
        input: List[EvalChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """
        Generate model response using batch inference.

        This method queues the request for batch processing and waits for the result.
        The actual inference is performed asynchronously in a background thread.

        Args:
            input: List of chat messages forming the conversation
            tools: Available tools for function calling (if supported)
            tool_choice: Tool selection strategy
            config: Generation configuration

        Returns:
            ModelOutput containing the generated response
        """
        # Synchronous mode avoids a background queue/thread and reduces lifecycle complexity.
        if self.eval_sync_mode:
            ms_input = convert_request(input, tools)
            ms_config = convert_config(config)
            completion = self.engine.infer([ms_input], ms_config, use_tqdm=False)[0]
            return self._to_evalscope_output(completion)

        # Ensure the background batch processing thread is running
        global batch_thread
        if batch_thread is None:
            batch_thread = Thread(target=_process_batches, daemon=True)
            batch_thread.start()

        # Convert EvalScope format to ms-swift format
        ms_input = convert_request(input, tools)
        ms_config = convert_config(config)

        # Package the request for batch processing
        batch_input = BatchInferInput(
            ms_input=ms_input, ms_config=ms_config, batch_size=config.batch_size, engine=self.engine)

        # Create a future to receive the result asynchronously
        future = Future[ModelOutput]()

        # Queue the request for batch processing
        batch_queue.put(_QueueItem(input=batch_input, future=future))

        # Block until the result is available
        return future.result()

    @staticmethod
    def _to_evalscope_output(completion) -> ModelOutput:
        choices = chat_choices_from_openai(completion, tools=[])
        usage = None
        if completion.usage:
            usage = ModelUsage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            )
        return ModelOutput(model=completion.model, choices=choices, usage=usage)


def _process_batches() -> None:
    """
    Background thread function that processes batched inference requests.

    This function runs continuously, collecting requests from the queue and
    processing them in batches for improved efficiency. It uses a timeout-based
    approach to balance between batch size and latency.
    """
    while True:
        # Collect requests from the queue until timeout or batch size limit
        inputs: List[Tuple[BatchInferInput, Future[ModelOutput]]] = []

        while True:
            try:
                # Wait for new requests with a 2-second timeout
                item = batch_queue.get(timeout=2)
                inputs.append((item.input, item.future))

                # Check if we've reached the desired batch size
                if len(inputs) == item.input.batch_size:
                    break  # Process this batch now

            except Empty:
                # No more requests in queue, process what we have
                break

        # Skip processing if no requests were collected
        if len(inputs) == 0:
            continue

        try:
            # Prepare batch inputs for ms-swift inference
            ms_inputs = [item[0].ms_input for item in inputs]
            ms_config = inputs[0][0].ms_config  # use first config for the batch
            engine = inputs[0][0].engine  # use first engine for the batch

            # Perform batch inference using ms-swift engine
            completions = engine.infer(ms_inputs, ms_config, use_tqdm=False)

            # Process results and deliver them to waiting futures
            for i, (batch_input, future) in enumerate(inputs):
                completion = completions[i]

                # Convert ms-swift response to EvalScope format
                result = EvalModel._to_evalscope_output(completion)

                # Deliver the result to the waiting caller
                future.set_result(result)

        except Exception as ex:
            # If batch processing fails, propagate the error to all waiting futures
            for _, future in inputs:
                future.set_exception(ex)


def convert_config(config: GenerateConfig) -> RequestConfig:
    """
    Convert EvalScope GenerateConfig to ms-swift RequestConfig.

    Maps configuration parameters between the two frameworks, ensuring
    compatibility while maintaining the same generation behavior.

    Args:
        config: EvalScope generation configuration

    Returns:
        RequestConfig: ms-swift compatible configuration
    """
    return RequestConfig(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        presence_penalty=(0.0 if config.presence_penalty is None else config.presence_penalty),
        frequency_penalty=(0.0 if config.frequency_penalty is None else config.frequency_penalty),
        seed=config.seed,
        stream=False,  # batch processing doesn't support streaming
        logprobs=config.logprobs,
        top_logprobs=config.top_logprobs)


def convert_request(messages: List[EvalChatMessage], tools: List[ToolInfo]) -> InferRequest:
    """
    Convert EvalScope request format to ms-swift InferRequest format.

    Transforms the message and tool format from EvalScope's representation
    to the format expected by ms-swift's inference engine.

    Args:
        messages: List of chat messages in EvalScope format
        tools: List of available tools in EvalScope format

    Returns:
        InferRequest: ms-swift compatible request object
    """
    # Convert tools to ms-swift format
    tools_list = []
    if len(tools) > 0:
        tools_list = [tool.model_dump(exclude_none=True) for tool in tools]

    # Convert messages to ms-swift format
    ms_messages = []
    for message in messages:
        ms_messages.append(message.model_dump(exclude_none=True))

    return InferRequest(
        messages=ms_messages,
        tools=tools_list,
    )
