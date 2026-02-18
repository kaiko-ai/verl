# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import contextlib
import functools
import inspect
import os
from contextvars import ContextVar
from typing import Optional

_trace_enabled: ContextVar[bool] = ContextVar("_trace_enabled", default=True)
_root_trace_span: ContextVar = ContextVar("_root_trace_span", default=None)


class RolloutTraceConfig:
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
    tracing backends like Weave and MLflow.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', 'arize', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        max_samples_per_step_per_worker (Optional[int]): Maximum number of unique samples to trace
            per worker per step. If None, all samples are traced. If set, each worker will randomly
            select up to this many unique samples to trace (including all their rollouts for GRPO).
            Total traces = max_samples_per_step_per_worker * num_workers * n_rollouts_per_sample.
        trace_step_interval (int): Only trace every N steps. E.g. 5 means trace on steps 0, 5, 10, ...
            Defaults to 1 (trace every step).
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: Optional[str] = None
    client: Optional[object] = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    max_samples_per_step_per_worker: Optional[int] = None
    trace_step_interval: int = 1
    arize_config: dict = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        backend: str,
        token2text: bool = False,
        max_samples_per_step_per_worker: Optional[int] = None,
        trace_step_interval: int = 1,
        arize_config: Optional[dict] = None,
    ):
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker
        config.trace_step_interval = max(1, trace_step_interval)
        config.arize_config = arize_config or {}

        if backend == "weave":
            import weave

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        elif backend == "arize":
            from arize.otel import register
            from opentelemetry import trace

            register(
                space_id=os.environ["ARIZE_SPACE_ID"],
                api_key=os.environ["ARIZE_API_KEY"],
                project_name=project_name,
            )
            config.client = trace.get_tracer(f"verl.rollout.{project_name}")
        else:
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> Optional[str]:
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> Optional[object]:
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> Optional[bool]:
        return cls.get_instance().token2text

    @classmethod
    def reset(cls):
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(
    sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False, trace: bool = True
):
    """A context manager to add attributes to a trace for the configured backend.

    Args:
        sample_index: Sample index for the trace.
        step: Training step number.
        rollout_n: Rollout number (for GRPO with multiple rollouts per sample).
        name: Name for the trace span (used by mlflow backend).
        validate: Whether this is a validation run.
        trace: If False, disables tracing for the duration of the context.
    """
    backend = RolloutTraceConfig.get_backend()

    should_skip = backend is not None and not trace

    if should_skip:
        token = _trace_enabled.set(False)
        try:
            yield
        finally:
            _trace_enabled.reset(token)
        return

    # Build attributes for the trace
    attributes = {}
    if backend:
        if sample_index is not None:
            attributes["sample_index"] = sample_index
        if step is not None:
            attributes["step"] = step
        if rollout_n is not None:
            attributes["rollout_n"] = rollout_n
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name

    if not attributes or backend is None:
        yield
        return

    if backend == "weave":
        import weave

        with weave.attributes(attributes):
            yield
    elif backend == "mlflow":
        import mlflow

        with mlflow.start_span(name=name) as span:
            trace_id = span.trace_id
            for key, value in attributes.items():
                mlflow.set_trace_tag(trace_id, str(key), str(value))
            yield
    elif backend == "arize":
        tracer = RolloutTraceConfig.get_client()
        span_attributes = {str(k): str(v) for k, v in attributes.items()}
        span_attributes["openinference.span.kind"] = "CHAIN"
        with tracer.start_as_current_span(
            name=name,
            attributes=span_attributes,
        ) as span:
            token = _root_trace_span.set(span)
            try:
                yield
            finally:
                _root_trace_span.reset(token)
    else:
        yield


def rollout_trace_op(func):
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return await func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        if backend is None:
            return await func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        async def add_token2text(self, result):
            if hasattr(result, "prompt_ids") and hasattr(self, "tokenizer") and hasattr(self.tokenizer, "decode"):
                _result = vars(result)
                loop = asyncio.get_running_loop()
                if hasattr(result, "prompt_ids"):
                    prompt_text = await loop.run_in_executor(None, self.tokenizer.decode, result.prompt_ids)
                    _result["prompt_text"] = prompt_text

                if hasattr(result, "response_ids"):
                    response_text = await loop.run_in_executor(None, self.tokenizer.decode, result.response_ids)
                    _result["response_text"] = response_text
                return _result
            return result

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                if enable_token2text:
                    _result = await add_token2text(self, result)
                    tracer.finish_call(call, output=_result)
                else:
                    tracer.finish_call(call, output=result)

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                span.set_inputs(inputs)
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    span.set_outputs(_result)
                else:
                    span.set_outputs(result)

            return result

        elif backend == "arize":
            from opentelemetry import trace as otel_trace

            # Flatten **kwargs into top-level so _set_arize_root_span_metadata sees them
            flat_inputs = {k: v for k, v in inputs.items() if k != "kwargs"}
            if isinstance(inputs.get("kwargs"), dict):
                flat_inputs.update(inputs["kwargs"])
            _set_arize_root_span_metadata(flat_inputs)

            tracer = RolloutTraceConfig.get_client()
            with tracer.start_as_current_span(
                name=func.__qualname__,
                attributes={"openinference.span.kind": "CHAIN"},
            ) as span:
                try:
                    result = await func(self, *args, **kwargs)
                    _set_arize_root_span_result(result)
                    _auto_attach_trace_conversation(result)
                    if enable_token2text:
                        _result = await add_token2text(self, result)
                        if isinstance(_result, dict) and "response_text" in _result:
                            span.set_attribute("output.value", _result["response_text"])
                            span.set_attribute("output.mime_type", "text/plain")
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                tracer.finish_call(call, output=result)
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        elif backend == "arize":
            from opentelemetry import trace as otel_trace

            flat_inputs = {k: v for k, v in inputs.items() if k != "kwargs"}
            if isinstance(inputs.get("kwargs"), dict):
                flat_inputs.update(inputs["kwargs"])
            _set_arize_root_span_metadata(flat_inputs)

            tracer = RolloutTraceConfig.get_client()
            with tracer.start_as_current_span(
                name=func.__qualname__,
                attributes={"openinference.span.kind": "CHAIN"},
            ) as span:
                try:
                    result = func(self, *args, **kwargs)
                    _set_arize_root_span_result(result)
                    _auto_attach_trace_conversation(result)
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper


def _set_arize_root_span_metadata(inputs: dict) -> None:
    """Extract scalar fields from function inputs and set them on the root trace span.

    For the Arize backend, writes flat top-level attributes on the ``rollout_trace_attr``
    span so they appear alongside ``step``, ``sample_index``, etc. and are searchable.
    Flattens one level of ``extra_info``. Skips complex types and keys already handled
    elsewhere (``messages``, ``multi_modal_inputs``, ``sampling_params``).
    """
    root_span = _root_trace_span.get()
    if root_span is None or not root_span.is_recording():
        return

    skip_keys = {"messages", "multi_modal_inputs", "sampling_params"}
    for key, value in inputs.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            root_span.set_attribute(key, str(value))
        elif isinstance(value, dict) and key == "extra_info":
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)):
                    root_span.set_attribute(f"extra_info.{sub_key}", str(sub_value))


def _set_arize_root_span_result(result) -> None:
    """Extract scalar fields from the function result and set them on the root trace span.

    Mirrors how Weave captures the full output object — but for Arize we only write
    scalar attributes (``reward_score``, ``num_turns``, etc.) to keep spans clean.
    """
    root_span = _root_trace_span.get()
    if root_span is None or not root_span.is_recording():
        return

    result_dict = vars(result) if hasattr(result, "__dict__") else {}
    skip_keys = {"prompt_ids", "response_ids", "response_mask", "response_logprobs",
                 "multi_modal_data", "extra_fields", "metrics"}
    for key, value in result_dict.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            root_span.set_attribute(key, str(value))


def _auto_attach_trace_conversation(result) -> None:
    """If result has trace_conversation, attach it to the current span and clear it."""
    messages = getattr(result, "trace_conversation", None)
    if messages is None:
        return
    images = None
    multi_modal = getattr(result, "multi_modal_data", None)
    if isinstance(multi_modal, dict):
        images = multi_modal.get("image")
    arize = RolloutTraceConfig.get_instance().arize_config
    rollout_trace_attach_conversation(
        messages=messages,
        images=images,
        image_format=arize.get("image_format", "png"),
        max_dimension=arize.get("max_dimension"),
        span_kind="AGENT",
    )
    result.trace_conversation = None


def _encode_image(img, image_format: str = "png", max_dimension: int | None = None) -> str | None:
    """Encode a PIL image to a base64 data URI. Returns None if img is not a valid image."""
    if not hasattr(img, "save"):
        return None

    import base64
    import io

    if max_dimension is not None:
        w, h = img.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            img = img.resize(
                (int(w * scale), int(h * scale)),
                resample=getattr(img, "Resampling", img).LANCZOS if hasattr(img, "Resampling") else 1,
            )

    buffer = io.BytesIO()
    img.save(buffer, format=image_format.upper())
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{b64}"


def rollout_trace_attach_conversation(
    messages: list[dict] | None,
    images: list | None = None,
    image_format: str = "png",
    max_dimension: int | None = None,
    span_kind: str | None = None,
):
    """Attach a conversation (messages + images) to the current trace span.

    For the Arize backend, serializes the conversation as OpenInference
    ``llm.input_messages`` attributes with roles, text, and inline images.
    Image placeholders (``{"type": "image"}``) in message content are resolved
    to base64-encoded images from the ``images`` list, consumed in order.

    Should be called once at the end of a ``@rollout_trace_op``-decorated function.

    For Weave/MLflow backends, this is a no-op.

    Args:
        messages: Chat message dicts with ``role`` and ``content`` keys.
            Content can be a string or a list of content items
            (``{"type": "text", "text": "..."}`` or ``{"type": "image"}``).
        images: Flat list of PIL Image objects, consumed in order as
            ``{"type": "image"}`` placeholders are encountered.
        image_format: Image encoding format ('png' or 'jpeg'). Default 'png'.
        max_dimension: If set, resize images so the largest dimension is at most
            this value (preserving aspect ratio).
        span_kind: If set, override the ``openinference.span.kind`` attribute
            on the current span (e.g. "AGENT").
    """
    if not messages:
        return

    backend = RolloutTraceConfig.get_backend()
    if backend != "arize":
        return

    if not _trace_enabled.get():
        return

    from opentelemetry import trace

    span = trace.get_current_span()
    if not span.is_recording():
        return

    if span_kind:
        span.set_attribute("openinference.span.kind", span_kind)

    image_idx = 0
    images = images or []

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content")
        prefix = f"llm.input_messages.{msg_idx}.message"

        # Normalize OmegaConf ListConfig to native Python list
        if not isinstance(content, (str, list, type(None))) and hasattr(content, "__iter__"):
            content = list(content)

        span.set_attribute(f"{prefix}.role", role)

        if isinstance(content, str):
            span.set_attribute(f"{prefix}.content", content)
        elif isinstance(content, list):
            has_images = any(
                hasattr(item, "get") and item.get("type") == "image" for item in content
            )
            if not has_images:
                # Text-only list: set as plain string to avoid duplication
                text = "\n".join(
                    item.get("text", "") for item in content if hasattr(item, "get") and item.get("type") == "text"
                )
                if text:
                    span.set_attribute(f"{prefix}.content", text)
            else:
                # Mixed content (text + images): use structured format only
                content_idx = 0
                for item in content:
                    item_type = item.get("type") if hasattr(item, "get") else None
                    content_prefix = f"{prefix}.contents.{content_idx}.message_content"

                    if item_type == "text":
                        text = item.get("text", "")
                        span.set_attribute(f"{content_prefix}.type", "text")
                        span.set_attribute(f"{content_prefix}.text", text)
                        content_idx += 1
                    elif item_type == "image":
                        if image_idx < len(images):
                            data_uri = _encode_image(images[image_idx], image_format, max_dimension)
                            if data_uri:
                                span.set_attribute(f"{content_prefix}.type", "image")
                                span.set_attribute(f"{content_prefix}.image.image.url", data_uri)
                                content_idx += 1
                        image_idx += 1
