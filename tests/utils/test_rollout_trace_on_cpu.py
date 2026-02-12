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

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attach_conversation, rollout_trace_attr, rollout_trace_op


@pytest.fixture(autouse=True)
def reset_rollout_trace_config_singleton():
    """Fixture to reset the RolloutTraceConfig singleton before each test."""
    RolloutTraceConfig.reset()


@pytest.fixture
def mock_weave_client():
    """Mocks the weave module and its client, yielding the mock client."""
    mock_weave = MagicMock()
    mock_client = MagicMock()
    mock_call = MagicMock()
    mock_client.create_call.return_value = mock_call
    mock_weave.init.return_value = mock_client

    # Also mock the call_context if it's used internally by the decorator
    mock_weave.trace.context.call_context.return_value = MagicMock()

    with patch.dict(sys.modules, {"weave": mock_weave, "weave.trace.context": mock_weave.trace.context}):
        yield mock_client


class TracedClass:
    @rollout_trace_op
    # @weave.op
    # @mlflow.trace
    async def my_method(self, a, b="default"):
        return f"result: {a}, {b}"

    @rollout_trace_op
    # @weave.op
    # @mlflow.trace
    async def middle_method(self, a, b="default"):
        await self.my_method("test_a1", b="test_b1")
        return f"result: {a}, {b}"

    @rollout_trace_op
    # @mlflow.trace
    async def my_method_with_exception(self):
        raise ValueError("Test Exception")

    async def upper_method(self):
        await self.my_method("test_a0", b="test_b0")
        await self.middle_method("test_a2", b="test_b2")
        return True


class UntracedClass:
    @rollout_trace_op
    async def my_method(self, x):
        return x * 2


async def test_rollout_trace_on_untraced_class():
    """Tests that the decorator works correctly when no backend is configured."""
    instance = UntracedClass()
    assert await instance.my_method(10) == 20


async def test_rollout_trace_with_tracer(mock_weave_client):
    """Tests that the decorator calls the tracer's methods correctly."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    instance = TracedClass()
    assert RolloutTraceConfig.get_client() is mock_weave_client

    result = await instance.my_method("test_a", b="test_b")

    assert result == "result: test_a, test_b"
    mock_weave_client.create_call.assert_called_once()
    call_kwargs = mock_weave_client.create_call.call_args.kwargs
    assert call_kwargs["op"] == "TracedClass.my_method"
    expected_inputs = {"a": "test_a", "b": "test_b"}
    assert call_kwargs["inputs"] == expected_inputs

    mock_call = mock_weave_client.create_call.return_value
    mock_weave_client.finish_call.assert_called_once_with(mock_call, output=result)


async def test_rollout_trace_with_exception(mock_weave_client):
    """Tests that `finish` is called with the exception when one is raised."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    instance = TracedClass()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    mock_weave_client.create_call.assert_called_once()
    mock_call = mock_weave_client.create_call.return_value
    mock_weave_client.finish_call.assert_called_once()

    # Check that finish_call was called with the exception
    args, kwargs = mock_weave_client.finish_call.call_args
    assert args[0] == mock_call
    assert "exception" in kwargs
    assert isinstance(kwargs["exception"], ValueError)


async def test_rollout_trace_with_dummy_backend(mock_weave_client):
    """Tests that the tracer is not called when the backend is 'dummy'."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="dummy")
    instance = TracedClass()

    await instance.my_method("test_a")

    mock_weave_client.create_call.assert_not_called()


async def test_trace_disabled_with_trace_false(mock_weave_client):
    """Tests that tracing is disabled when trace=False."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    assert RolloutTraceConfig.get_backend() == "weave"

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    # No tracing should have occurred
    mock_weave_client.create_call.assert_not_called()

    # Verify that tracing works again with trace=True (default)
    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


async def test_trace_false_disables_nested_trace_ops(mock_weave_client):
    """Tests that trace=False disables all nested @rollout_trace_op calls."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        # Call upper_method which internally calls my_method and middle_method
        # All of these are decorated with @rollout_trace_op
        result = await instance.upper_method()
        assert result is True

    # No tracing should have occurred for any of the nested calls
    mock_weave_client.create_call.assert_not_called()

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


async def test_trace_enabled_restored_after_exception(mock_weave_client):
    """Tests that trace state is restored even if an exception occurs when trace=False."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    assert RolloutTraceConfig.get_backend() == "weave"

    # Use trace=False and raise an exception
    try:
        with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
            raise RuntimeError("Test exception with trace disabled")
    except RuntimeError:
        pass

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


@pytest.mark.skipif(
    os.environ.get("RUN_WEAVE_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping weave integration test. Set RUN_WEAVE_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_weave_backend():
    """Integration test with a real weave backend."""

    # This assumes that the weave environment (e.g., project) is configured
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3):
        await instance.upper_method()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    print("\nWeave integration test ran successfully. Check your weave project for the trace.")


@pytest.mark.skipif(
    os.environ.get("RUN_MLFLOW_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping mlflow integration test. Set RUN_MLFLOW_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_mlflow_backend():
    """Integration test with a real mlflow backend."""

    # This assumes that the mlflow environment (e.g., project) is configured
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="mlflow")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3, name="agent_run"):
        assert await instance.upper_method()

    # with pytest.raises(ValueError, match="Test Exception"):
    #     await instance.my_method_with_exception()

    print("\nMlflow integration test ran successfully. Check your mlflow project for the trace.")


@pytest.fixture
def mock_arize_tracer():
    """Mocks the arize.otel and opentelemetry modules, yielding (mock_tracer, mock_span)."""
    mock_arize_otel = MagicMock()
    mock_otel = MagicMock()
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer.start_as_current_span.return_value = mock_span
    mock_otel.trace.get_tracer.return_value = mock_tracer
    mock_otel.trace.Status = MagicMock()
    mock_otel.trace.StatusCode = MagicMock()
    mock_otel.trace.StatusCode.OK = "OK"
    mock_otel.trace.StatusCode.ERROR = "ERROR"

    with patch.dict(
        sys.modules,
        {
            "arize": MagicMock(),
            "arize.otel": mock_arize_otel,
            "opentelemetry": mock_otel,
            "opentelemetry.trace": mock_otel.trace,
        },
    ):
        with patch.dict(os.environ, {"ARIZE_SPACE_ID": "test-space", "ARIZE_API_KEY": "test-key"}):
            yield mock_tracer, mock_span


async def test_rollout_trace_with_arize_tracer(mock_arize_tracer):
    """Tests that the Arize backend creates spans without input/output dumps."""
    mock_tracer, mock_span = mock_arize_tracer
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")
    instance = TracedClass()

    result = await instance.my_method("test_a", b="test_b")

    assert result == "result: test_a, test_b"
    mock_tracer.start_as_current_span.assert_called_once()
    call_kwargs = mock_tracer.start_as_current_span.call_args.kwargs
    assert call_kwargs["name"] == "TracedClass.my_method"
    # Child span should only have span kind, not input/output dumps
    assert call_kwargs["attributes"] == {"openinference.span.kind": "CHAIN"}
    # No output.value with raw token dump
    output_calls = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == "output.value"]
    assert len(output_calls) == 0


async def test_rollout_trace_arize_with_exception(mock_arize_tracer):
    """Tests that the Arize backend records exceptions on spans."""
    mock_tracer, mock_span = mock_arize_tracer
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")
    instance = TracedClass()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    mock_tracer.start_as_current_span.assert_called_once()
    mock_span.record_exception.assert_called_once()
    assert isinstance(mock_span.record_exception.call_args[0][0], ValueError)
    mock_span.set_status.assert_called_once()


async def test_trace_disabled_with_arize(mock_arize_tracer):
    """Tests that trace=False disables Arize tracing."""
    mock_tracer, mock_span = mock_arize_tracer
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")
    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    mock_tracer.start_as_current_span.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("RUN_ARIZE_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping Arize integration test. Set RUN_ARIZE_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_arize_backend():
    """Integration test with a real Arize backend."""

    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3):
        await instance.upper_method()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    print("\nArize integration test ran successfully. Check your Arize project for the trace.")


class FakePILImage:
    """Minimal duck-type stand-in for a PIL Image."""

    def __init__(self, width=100, height=80):
        self._size = (width, height)

    @property
    def size(self):
        return self._size

    def resize(self, size, resample=None):
        return FakePILImage(width=size[0], height=size[1])

    def save(self, fp, format=None):
        # Write a tiny valid payload so base64 is non-empty
        fp.write(b"\x89PNG_FAKE")

    class Resampling:
        LANCZOS = 1


def _setup_mock_span(mock_arize_tracer):
    """Helper to configure mock span for attach_conversation tests."""
    import sys

    mock_tracer, mock_span = mock_arize_tracer
    mock_span.is_recording.return_value = True
    mock_otel = sys.modules["opentelemetry"]
    mock_otel.trace.get_current_span.return_value = mock_span
    return mock_span


async def test_attach_conversation_structured_messages(mock_arize_tracer):
    """Tests that a multi-turn conversation is serialized with roles, text, and images."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Analyze"}, {"type": "image"}]},
        {"role": "assistant", "content": "I see tissue."},
        {"role": "tool", "content": [{"type": "text", "text": "Patch result"}, {"type": "image"}]},
    ]
    images = [FakePILImage(), FakePILImage()]
    rollout_trace_attach_conversation(messages, images, image_format="png")

    calls = {args[0]: args[1] for args, _ in mock_span.set_attribute.call_args_list}

    # Message 0: system with string content
    assert calls["llm.input_messages.0.message.role"] == "system"
    assert calls["llm.input_messages.0.message.content"] == "You are helpful."

    # Message 1: user with text + image (mixed → structured only, no message.content)
    assert calls["llm.input_messages.1.message.role"] == "user"
    assert "llm.input_messages.1.message.content" not in calls
    assert calls["llm.input_messages.1.message.contents.0.message_content.type"] == "text"
    assert calls["llm.input_messages.1.message.contents.0.message_content.text"] == "Analyze"
    assert calls["llm.input_messages.1.message.contents.1.message_content.type"] == "image"
    assert calls["llm.input_messages.1.message.contents.1.message_content.image.image.url"].startswith(
        "data:image/png;base64,"
    )

    # Message 2: assistant with string content
    assert calls["llm.input_messages.2.message.role"] == "assistant"
    assert calls["llm.input_messages.2.message.content"] == "I see tissue."

    # Message 3: tool with text + image (mixed → structured only, no message.content)
    assert calls["llm.input_messages.3.message.role"] == "tool"
    assert "llm.input_messages.3.message.content" not in calls
    assert calls["llm.input_messages.3.message.contents.0.message_content.type"] == "text"
    assert calls["llm.input_messages.3.message.contents.1.message_content.type"] == "image"


async def test_attach_conversation_sets_span_kind(mock_arize_tracer):
    """Tests that span_kind parameter overrides openinference.span.kind attribute."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    messages = [{"role": "user", "content": "hello"}]
    rollout_trace_attach_conversation(messages, span_kind="AGENT")

    calls = {args[0]: args[1] for args, _ in mock_span.set_attribute.call_args_list}
    assert calls["openinference.span.kind"] == "AGENT"


async def test_attach_conversation_noop_without_backend():
    """Tests that attach_conversation is a no-op when no backend is configured."""
    messages = [{"role": "user", "content": "hello"}]
    rollout_trace_attach_conversation(messages, [FakePILImage()])


async def test_attach_conversation_noop_with_weave(mock_weave_client):
    """Tests that attach_conversation is a no-op when Weave backend is configured."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    messages = [{"role": "user", "content": "hello"}]
    rollout_trace_attach_conversation(messages, [FakePILImage()])


async def test_attach_conversation_with_resize(mock_arize_tracer):
    """Tests that images are resized when max_dimension is set."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    saved_sizes = []

    class TrackingImage(FakePILImage):
        def resize(self, size, resample=None):
            return TrackingImage(width=size[0], height=size[1])

        def save(self, fp, format=None):
            saved_sizes.append(self.size)
            fp.write(b"\x89PNG_FAKE")

    messages = [{"role": "user", "content": [{"type": "image"}]}]
    rollout_trace_attach_conversation(messages, [TrackingImage(width=1024, height=512)], max_dimension=256)

    assert saved_sizes[0] == (256, 128)


async def test_attach_conversation_respects_trace_disabled(mock_arize_tracer):
    """Tests that attach_conversation is a no-op when tracing is disabled."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="arize")

    messages = [{"role": "user", "content": [{"type": "image"}]}]

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        rollout_trace_attach_conversation(messages, [FakePILImage()])

    mock_span.set_attribute.assert_not_called()


# ============================================================================
# Diagnostic tests — verify attribute correctness before deploying
# ============================================================================

def _build_realistic_kaiko_messages():
    """Build a realistic multi-turn conversation matching kaiko_tool_agent output."""
    return [
        {
            "role": "system",
            "content": "You are a slide-level analysis agent. Use tools to examine regions.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this slide for abnormal tissue."},
                {"type": "image"},
            ],
        },
        # After first generation — assistant message is a string (decoded tokens)
        {"role": "assistant", "content": "I'll examine the tissue. Let me use extract_patch."},
        # Tool response with image
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Extracted patch at (100, 200), size 512x512"},
                {"type": "image"},
            ],
        },
        # Tool response with multiple images
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Segmentation results for region A"},
                {"type": "image"},
                {"type": "image"},
            ],
        },
        # Tool response without images — plain string
        {"role": "tool", "content": "Invalid tool call: missing argument 'x'"},
        # Last-turn message
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is the last turn, please make the final analysis.",
                }
            ],
        },
    ]


def _collect_attributes(mock_span):
    """Collect all set_attribute calls as a dict."""
    return {args[0]: args[1] for args, _ in mock_span.set_attribute.call_args_list}


async def test_diagnostic_realistic_kaiko_conversation(mock_arize_tracer):
    """Full realistic multi-turn conversation — dumps all attributes for inspection."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="p", experiment_name="e", backend="arize")

    messages = _build_realistic_kaiko_messages()
    images = [FakePILImage(w, h) for w, h in [(512, 512), (512, 512), (256, 256), (256, 256)]]

    rollout_trace_attach_conversation(
        messages, images, image_format="jpeg", max_dimension=512, span_kind="AGENT"
    )

    attrs = _collect_attributes(mock_span)

    # --- Span kind ---
    assert attrs["openinference.span.kind"] == "AGENT"

    # --- Message 0: system (string content) ---
    assert attrs["llm.input_messages.0.message.role"] == "system"
    assert attrs["llm.input_messages.0.message.content"] == (
        "You are a slide-level analysis agent. Use tools to examine regions."
    )

    # --- Message 1: user with text + 1 image (mixed → structured only) ---
    assert attrs["llm.input_messages.1.message.role"] == "user"
    assert "llm.input_messages.1.message.content" not in attrs
    assert attrs["llm.input_messages.1.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.1.message.contents.0.message_content.text"] == (
        "Analyze this slide for abnormal tissue."
    )
    assert attrs["llm.input_messages.1.message.contents.1.message_content.type"] == "image"
    img_url = attrs["llm.input_messages.1.message.contents.1.message_content.image.image.url"]
    assert img_url.startswith("data:image/jpeg;base64,"), f"Bad image URL prefix: {img_url[:40]}"

    # --- Message 2: assistant (string content) ---
    assert attrs["llm.input_messages.2.message.role"] == "assistant"
    assert attrs["llm.input_messages.2.message.content"] == (
        "I'll examine the tissue. Let me use extract_patch."
    )

    # --- Message 3: tool with text + 1 image (mixed → structured only) ---
    assert attrs["llm.input_messages.3.message.role"] == "tool"
    assert "llm.input_messages.3.message.content" not in attrs
    assert attrs["llm.input_messages.3.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.3.message.contents.0.message_content.text"] == (
        "Extracted patch at (100, 200), size 512x512"
    )
    assert attrs["llm.input_messages.3.message.contents.1.message_content.type"] == "image"

    # --- Message 4: tool with text + 2 images (mixed → structured only) ---
    assert attrs["llm.input_messages.4.message.role"] == "tool"
    assert "llm.input_messages.4.message.content" not in attrs
    assert attrs["llm.input_messages.4.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.4.message.contents.1.message_content.type"] == "image"
    assert attrs["llm.input_messages.4.message.contents.2.message_content.type"] == "image"

    # --- Message 5: tool (plain string, no images) ---
    assert attrs["llm.input_messages.5.message.role"] == "tool"
    assert attrs["llm.input_messages.5.message.content"] == (
        "Invalid tool call: missing argument 'x'"
    )

    # --- Message 6: user text-only list content (text-only → message.content only) ---
    assert attrs["llm.input_messages.6.message.role"] == "user"
    assert "llm.input_messages.6.message.contents.0.message_content.type" not in attrs
    assert attrs["llm.input_messages.6.message.content"] == (
        "This is the last turn, please make the final analysis."
    )

    # All 4 images consumed
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]
    assert len(image_url_keys) == 4, f"Expected 4 image URLs, got {len(image_url_keys)}: {image_url_keys}"


async def test_diagnostic_list_content_mixed_uses_structured_only(mock_arize_tracer):
    """Verifies that mixed list content (text+images) uses structured format only.

    To avoid double-rendering in Arize, mixed content only sets message.contents.*
    (not message.content). Text-only lists set message.content only.
    """
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="p", experiment_name="e", backend="arize")

    messages = [
        {"role": "tool", "content": [
            {"type": "text", "text": "Patch at (100, 200)"},
            {"type": "image"},
            {"type": "text", "text": "Region analysis complete"},
        ]},
    ]
    rollout_trace_attach_conversation(messages, [FakePILImage()])

    attrs = _collect_attributes(mock_span)

    # Role is set
    assert attrs["llm.input_messages.0.message.role"] == "tool"

    # Structured content is set (mixed → structured only)
    assert attrs["llm.input_messages.0.message.contents.0.message_content.type"] == "text"
    assert attrs["llm.input_messages.0.message.contents.0.message_content.text"] == "Patch at (100, 200)"
    assert attrs["llm.input_messages.0.message.contents.1.message_content.type"] == "image"
    assert attrs["llm.input_messages.0.message.contents.2.message_content.type"] == "text"

    # message.content is NOT set for mixed content (avoids double-rendering)
    assert "llm.input_messages.0.message.content" not in attrs


async def test_diagnostic_image_count_mismatch_fewer_images(mock_arize_tracer):
    """Tests behavior when fewer images than placeholders — images silently skipped."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="p", experiment_name="e", backend="arize")

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "image"}]},
    ]
    # Only 1 image for 3 placeholders
    rollout_trace_attach_conversation(messages, [FakePILImage()])

    attrs = _collect_attributes(mock_span)
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]

    # Only 1 image attached, 2 placeholders silently skipped
    assert len(image_url_keys) == 1, f"Expected 1 image, got {len(image_url_keys)}"


async def test_diagnostic_image_count_mismatch_more_images(mock_arize_tracer):
    """Tests behavior when more images than placeholders — extra images ignored."""
    mock_span = _setup_mock_span(mock_arize_tracer)
    RolloutTraceConfig.init(project_name="p", experiment_name="e", backend="arize")

    messages = [
        {"role": "user", "content": [{"type": "image"}]},
    ]
    # 3 images for 1 placeholder
    rollout_trace_attach_conversation(messages, [FakePILImage(), FakePILImage(), FakePILImage()])

    attrs = _collect_attributes(mock_span)
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]

    # Only 1 image attached (matching the placeholder)
    assert len(image_url_keys) == 1, f"Expected 1 image, got {len(image_url_keys)}"


async def test_diagnostic_base64_image_size():
    """Measures actual base64 image size to verify it fits within OPENINFERENCE limits."""
    from verl.utils.rollout_trace import _encode_image

    class RealisticImage(FakePILImage):
        """Writes a payload similar in size to a 512x512 JPEG (~30-80KB)."""

        def save(self, fp, format=None):
            # Simulate ~50KB JPEG
            fp.write(b"\xff\xd8" + b"\x00" * 50_000 + b"\xff\xd9")

    data_uri = _encode_image(RealisticImage(512, 512), image_format="jpeg")
    assert data_uri is not None

    # base64 of 50KB ≈ 66K chars + prefix
    uri_len = len(data_uri)
    prefix = "data:image/jpeg;base64,"
    assert data_uri.startswith(prefix)

    # Should be well under 200K (our env var limit)
    assert uri_len < 200_000, f"Image URI is {uri_len} chars — exceeds 200K limit!"
    # Should exceed old default 32K limit (this was the REDACTED bug)
    assert uri_len > 32_000, f"Image URI is {uri_len} chars — under old 32K default"

    print(f"\n  Base64 image URI length: {uri_len:,} chars (limit: 200,000)")


async def test_diagnostic_openinference_mask_passes_our_attributes():
    """Uses the REAL OpenInference TraceConfig.mask() to verify our keys aren't dropped.

    This catches cases where mask() returns None for our attribute keys,
    which would cause OpenInferenceSpan to silently drop them.
    """
    from openinference.instrumentation.config import TraceConfig

    # Default config (all hide_* = False), with our custom image max length
    config = TraceConfig(base64_image_max_length=200_000)

    test_keys_and_values = [
        ("openinference.span.kind", "AGENT"),
        ("llm.input_messages.0.message.role", "system"),
        ("llm.input_messages.0.message.content", "You are helpful."),
        ("llm.input_messages.1.message.role", "user"),
        ("llm.input_messages.1.message.contents.0.message_content.type", "text"),
        ("llm.input_messages.1.message.contents.0.message_content.text", "Analyze this"),
        ("llm.input_messages.1.message.contents.1.message_content.type", "image"),
        (
            "llm.input_messages.1.message.contents.1.message_content.image.image.url",
            "data:image/jpeg;base64,/9j/AAAA",  # short fake base64
        ),
        ("llm.input_messages.2.message.role", "tool"),
        ("llm.input_messages.2.message.contents.0.message_content.type", "text"),
        ("llm.input_messages.2.message.contents.0.message_content.text", "Patch result"),
    ]

    dropped = []
    for key, value in test_keys_and_values:
        masked = config.mask(key, value)
        if masked is None:
            dropped.append(key)

    assert not dropped, f"TraceConfig.mask() dropped these keys: {dropped}"


async def test_diagnostic_openinference_mask_redacts_large_images():
    """Verifies that the mask REDACTS images over the size limit."""
    from openinference.instrumentation.config import TraceConfig

    # With default 32K limit (simulating missing env var)
    config_default = TraceConfig(base64_image_max_length=32_000)

    # Simulate a 50KB image base64 (≈67K chars)
    import base64

    fake_data = b"\xff" * 50_000
    b64 = base64.b64encode(fake_data).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    key = "llm.input_messages.0.message.contents.0.message_content.image.image.url"
    masked = config_default.mask(key, data_uri)
    assert masked == "__REDACTED__", (
        f"Expected REDACTED with 32K limit, got: {str(masked)[:60]}..."
    )

    # With our 200K limit — should pass through
    config_custom = TraceConfig(base64_image_max_length=200_000)
    masked = config_custom.mask(key, data_uri)
    assert masked == data_uri, "Image should NOT be redacted with 200K limit"


async def test_diagnostic_openinference_mask_with_env_var():
    """Verifies that OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH env var is read by TraceConfig."""
    from openinference.instrumentation.config import TraceConfig

    import base64

    fake_data = b"\xff" * 50_000
    b64 = base64.b64encode(fake_data).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    key = "llm.input_messages.0.message.contents.0.message_content.image.image.url"

    # Without env var — uses default 32K, should REDACT
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH", None)
        config_no_env = TraceConfig()
        assert config_no_env.base64_image_max_length == 32_000
        assert config_no_env.mask(key, data_uri) == "__REDACTED__"

    # With env var set to 200K — should pass
    with patch.dict(os.environ, {"OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH": "200000"}):
        config_with_env = TraceConfig()
        assert config_with_env.base64_image_max_length == 200_000
        assert config_with_env.mask(key, data_uri) == data_uri


# ============================================================================
# Real pipeline tests — uses actual OTel SDK spans, not mocks
# ============================================================================

@pytest.fixture
def real_otel_pipeline():
    """Creates a real OpenInference tracing pipeline with in-memory span export.

    Yields (tracer, get_exported_spans) where:
    - tracer: a real OITracer that creates real OpenInferenceSpans
    - get_exported_spans: callable returning list of exported ReadableSpans

    Also sets RolloutTraceConfig.backend = "arize" so rollout_trace_attach_conversation works.
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.instrumentation.config import TraceConfig

    exporter = InMemorySpanExporter()

    config = TraceConfig(base64_image_max_length=200_000)
    provider = OITracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("test-tracer")

    # Set backend so rollout_trace_attach_conversation doesn't bail out
    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc._initialized = True

    def get_exported_spans():
        provider.force_flush()
        return exporter.get_finished_spans()

    yield tracer, get_exported_spans

    provider.shutdown()


def test_real_pipeline_attributes_land_on_span(real_otel_pipeline):
    """Uses REAL OTel SDK + OpenInference pipeline to verify attributes are exported.

    This catches issues that mock tests miss:
    - OTel rejecting attribute value types
    - OpenInferenceSpan masking dropping keys
    - trace.get_current_span() returning wrong span type
    - SpanLimits truncating attributes
    """
    from opentelemetry import trace

    tracer, get_spans = real_otel_pipeline

    messages = _build_realistic_kaiko_messages()
    images = [FakePILImage(512, 512) for _ in range(4)]

    # Simulate what rollout_trace_op does for arize backend
    with tracer.start_as_current_span(
        name="KaikoToolAgent.run",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": "test inputs",
            "input.mime_type": "text/plain",
        },
    ) as span:
        # This is what our code does inside the traced function
        current = trace.get_current_span()

        # Verify we get the right span type
        assert current is not None
        assert hasattr(current, 'set_attribute'), "get_current_span() returned invalid span"

        rollout_trace_attach_conversation(
            messages, images,
            image_format="jpeg", max_dimension=512, span_kind="AGENT",
        )

    exported = get_spans()
    assert len(exported) == 1, f"Expected 1 span, got {len(exported)}"

    attrs = dict(exported[0].attributes)

    # --- Verify span kind was overridden ---
    assert attrs.get("openinference.span.kind") == "AGENT", (
        f"span.kind = {attrs.get('openinference.span.kind')!r}, expected 'AGENT'"
    )

    # --- Message 0: system (string content) ---
    assert attrs.get("llm.input_messages.0.message.role") == "system", (
        f"Missing or wrong role for msg 0: {attrs.get('llm.input_messages.0.message.role')!r}"
    )
    assert "llm.input_messages.0.message.content" in attrs, (
        "message.content missing for system message (string content)"
    )

    # --- Message 1: user with list content [text, image] (mixed → structured only) ---
    assert attrs.get("llm.input_messages.1.message.role") == "user"
    assert "llm.input_messages.1.message.content" not in attrs
    # Structured contents
    assert attrs.get("llm.input_messages.1.message.contents.0.message_content.type") == "text"
    assert "llm.input_messages.1.message.contents.0.message_content.text" in attrs
    assert attrs.get("llm.input_messages.1.message.contents.1.message_content.type") == "image"
    img_key = "llm.input_messages.1.message.contents.1.message_content.image.image.url"
    assert img_key in attrs, f"Image URL not in exported span! Keys containing 'image': {[k for k in attrs if 'image' in k]}"
    assert attrs[img_key].startswith("data:image/jpeg;base64,"), (
        f"Image URL has wrong format: {attrs[img_key][:50]}..."
    )

    # --- Message 2: assistant (string content) ---
    assert attrs.get("llm.input_messages.2.message.role") == "assistant"
    assert attrs.get("llm.input_messages.2.message.content") == (
        "I'll examine the tissue. Let me use extract_patch."
    )

    # --- Message 3: tool with list content [text, image] (mixed → structured only) ---
    assert attrs.get("llm.input_messages.3.message.role") == "tool"
    assert "llm.input_messages.3.message.content" not in attrs
    assert attrs.get("llm.input_messages.3.message.contents.0.message_content.type") == "text"
    assert attrs.get("llm.input_messages.3.message.contents.1.message_content.type") == "image"

    # --- Message 5: tool with plain string content ---
    assert attrs.get("llm.input_messages.5.message.role") == "tool"
    assert attrs.get("llm.input_messages.5.message.content") == (
        "Invalid tool call: missing argument 'x'"
    )

    # --- Count total image URLs ---
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]
    assert len(image_url_keys) == 4, (
        f"Expected 4 images in exported span, got {len(image_url_keys)}: {image_url_keys}"
    )

    # --- Print summary for debugging ---
    all_msg_keys = sorted(k for k in attrs if k.startswith("llm.input_messages"))
    print(f"\n  Exported {len(all_msg_keys)} message attributes on real OTel span")
    for k in all_msg_keys:
        v = attrs[k]
        if isinstance(v, str) and len(v) > 80:
            v = v[:80] + "..."
        print(f"    {k} = {v!r}")


def test_real_pipeline_get_current_span_returns_oi_span(real_otel_pipeline):
    """Verifies trace.get_current_span() inside an OITracer span returns a usable span."""
    from opentelemetry import trace

    tracer, get_spans = real_otel_pipeline

    with tracer.start_as_current_span("test-span") as span:
        current = trace.get_current_span()

        # Must be recording (not INVALID_SPAN)
        assert current.is_recording(), "get_current_span() returned non-recording span!"

        # Must accept string attributes without error
        current.set_attribute("test.key", "test.value")

    exported = get_spans()
    assert len(exported) == 1
    assert exported[0].attributes.get("test.key") == "test.value"


def test_real_pipeline_image_not_redacted_with_200k_limit(real_otel_pipeline):
    """Verifies that a ~50KB image base64 is NOT redacted with 200K limit in real pipeline."""
    from opentelemetry import trace

    tracer, get_spans = real_otel_pipeline

    class BigImage(FakePILImage):
        def save(self, fp, format=None):
            fp.write(b"\xff\xd8" + b"\x00" * 50_000 + b"\xff\xd9")

    messages = [{"role": "user", "content": [{"type": "image"}]}]

    with tracer.start_as_current_span("test-span", attributes={"openinference.span.kind": "AGENT"}):
        rollout_trace_attach_conversation(messages, [BigImage(512, 512)], image_format="jpeg")

    exported = get_spans()
    attrs = dict(exported[0].attributes)

    img_key = "llm.input_messages.0.message.contents.0.message_content.image.image.url"
    assert img_key in attrs, f"Image key missing! All keys: {sorted(attrs.keys())}"
    assert attrs[img_key] != "__REDACTED__", "Image was REDACTED despite 200K limit!"
    assert attrs[img_key].startswith("data:image/jpeg;base64,")


def test_real_pipeline_image_redacted_with_default_limit():
    """Verifies that without custom limit, a ~50KB image IS redacted in real pipeline."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.instrumentation.config import TraceConfig

    exporter = InMemorySpanExporter()
    # Default config: base64_image_max_length = 32_000
    config = TraceConfig()
    provider = OITracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-default-limit")

    # Set backend so rollout_trace_attach_conversation works
    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc._initialized = True

    class BigImage(FakePILImage):
        def save(self, fp, format=None):
            fp.write(b"\xff\xd8" + b"\x00" * 50_000 + b"\xff\xd9")

    messages = [{"role": "user", "content": [{"type": "image"}]}]

    with tracer.start_as_current_span("test-span", attributes={"openinference.span.kind": "AGENT"}):
        rollout_trace_attach_conversation(messages, [BigImage(512, 512)], image_format="jpeg")

    provider.force_flush()
    exported = exporter.get_finished_spans()
    attrs = dict(exported[0].attributes)

    img_key = "llm.input_messages.0.message.contents.0.message_content.image.image.url"
    assert img_key in attrs, f"Image key missing! Keys: {sorted(attrs.keys())}"
    assert attrs[img_key] == "__REDACTED__", (
        f"Expected REDACTED with default 32K limit, got: {str(attrs[img_key])[:60]}..."
    )

    provider.shutdown()


# ============================================================================
# Full-flow test — rollout_trace_op + attach_conversation (mirrors production)
# ============================================================================


async def test_full_flow_rollout_trace_op_with_attach_conversation():
    """End-to-end test: @rollout_trace_op creates span, attach_conversation sets attrs.

    This mirrors the production flow in KaikoToolAgent.run():
    1. @rollout_trace_op decorator creates the span via tracer.start_as_current_span()
    2. Inside the function, rollout_trace_attach_conversation() is called
    3. Attributes should appear on the exported span

    Uses a real OITracerProvider (same as arize.otel.register() creates) with
    InMemorySpanExporter to capture and verify exported spans.
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.instrumentation.config import TraceConfig

    exporter = InMemorySpanExporter()
    config = TraceConfig(base64_image_max_length=200_000)
    provider = OITracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-full-flow")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True

    class AgentUnderTest:
        @rollout_trace_op
        async def run(self, messages, images):
            """Simulates KaikoToolAgent.run() — decorated, calls attach at end."""
            rollout_trace_attach_conversation(
                messages,
                images,
                image_format="jpeg",
                max_dimension=512,
                span_kind="AGENT",
            )
            return "done"

    messages = _build_realistic_kaiko_messages()
    images = [FakePILImage(512, 512) for _ in range(4)]

    agent = AgentUnderTest()
    result = await agent.run(messages, images)
    assert result == "done"

    provider.force_flush()
    exported = exporter.get_finished_spans()
    assert len(exported) >= 1, f"Expected at least 1 span, got {len(exported)}"

    span = exported[0]
    attrs = dict(span.attributes)

    # Verify span kind was overridden to AGENT
    assert attrs.get("openinference.span.kind") == "AGENT", (
        f"span.kind={attrs.get('openinference.span.kind')!r}, expected 'AGENT'"
    )

    # Verify system message (string content)
    assert attrs.get("llm.input_messages.0.message.role") == "system"
    assert "llm.input_messages.0.message.content" in attrs

    # Verify user message with list content [text, image] (mixed → structured only)
    assert attrs.get("llm.input_messages.1.message.role") == "user"
    assert "llm.input_messages.1.message.content" not in attrs
    assert attrs.get("llm.input_messages.1.message.contents.0.message_content.type") == "text"
    assert "llm.input_messages.1.message.contents.0.message_content.text" in attrs

    img_key = "llm.input_messages.1.message.contents.1.message_content.image.image.url"
    assert img_key in attrs, (
        f"Image URL missing! Keys with 'image': {[k for k in attrs if 'image' in k.lower()]}"
    )
    assert str(attrs[img_key]).startswith("data:image/jpeg;base64,"), (
        f"Image URL wrong format: {str(attrs[img_key])[:50]}"
    )
    assert attrs[img_key] != "__REDACTED__", "Image was REDACTED despite 200K limit!"

    # Verify assistant message (string content)
    assert attrs.get("llm.input_messages.2.message.role") == "assistant"
    assert "llm.input_messages.2.message.content" in attrs

    # Verify tool message with list content (mixed → structured only)
    assert attrs.get("llm.input_messages.3.message.role") == "tool"
    assert "llm.input_messages.3.message.content" not in attrs

    # Verify plain string tool message
    assert attrs.get("llm.input_messages.5.message.role") == "tool"
    assert attrs.get("llm.input_messages.5.message.content") == (
        "Invalid tool call: missing argument 'x'"
    )

    # Count total images
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]
    assert len(image_url_keys) == 4, (
        f"Expected 4 images, got {len(image_url_keys)}: {image_url_keys}"
    )

    # Print all llm.input_messages attributes for debugging
    msg_keys = sorted(k for k in attrs if k.startswith("llm.input_messages"))
    print(f"\n  Full-flow test: {len(msg_keys)} message attributes on exported span")
    for k in msg_keys:
        v = attrs[k]
        if isinstance(v, str) and len(v) > 100:
            v = v[:100] + "..."
        print(f"    {k} = {v!r}")

    provider.shutdown()


async def test_full_flow_with_omegaconf_listconfig():
    """Verifies that OmegaConf ListConfig/DictConfig content is handled correctly.

    In production, Hydra/OmegaConf wraps config values in ListConfig/DictConfig
    instead of native list/dict. This caused the original bug where isinstance(content, list)
    returned False for ListConfig.
    """
    from omegaconf import DictConfig, ListConfig
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider
    from openinference.instrumentation.config import TraceConfig

    exporter = InMemorySpanExporter()
    config = TraceConfig(base64_image_max_length=200_000)
    provider = OITracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-omegaconf")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True

    # Build messages using OmegaConf types (matching production)
    messages = [
        DictConfig({
            "role": "system",
            "content": ListConfig([
                DictConfig({"type": "text", "text": "You are a helpful agent."}),
            ]),
        }),
        DictConfig({
            "role": "user",
            "content": ListConfig([
                DictConfig({"type": "text", "text": "Analyze this slide."}),
                DictConfig({"type": "image"}),
            ]),
        }),
        DictConfig({
            "role": "assistant",
            "content": "I see tissue with abnormal patterns.",
        }),
        DictConfig({
            "role": "tool",
            "content": ListConfig([
                DictConfig({"type": "text", "text": "Patch at (100, 200)"}),
                DictConfig({"type": "image"}),
            ]),
        }),
        DictConfig({
            "role": "tool",
            "content": "Invalid tool call: missing argument",
        }),
    ]
    images = [FakePILImage(512, 512), FakePILImage(256, 256)]

    class AgentUnderTest:
        @rollout_trace_op
        async def run(self, msgs, imgs):
            rollout_trace_attach_conversation(
                msgs, imgs, image_format="jpeg", max_dimension=512, span_kind="AGENT",
            )
            return "done"

    agent = AgentUnderTest()
    await agent.run(messages, images)

    provider.force_flush()
    exported = exporter.get_finished_spans()
    assert len(exported) >= 1
    attrs = dict(exported[0].attributes)

    # System message with text-only ListConfig → message.content only
    assert attrs.get("llm.input_messages.0.message.role") == "system"
    assert attrs.get("llm.input_messages.0.message.content") == "You are a helpful agent."
    assert "llm.input_messages.0.message.contents.0.message_content.type" not in attrs

    # User message with ListConfig [text, image] → mixed → structured only
    assert attrs.get("llm.input_messages.1.message.role") == "user"
    assert "llm.input_messages.1.message.content" not in attrs
    assert attrs.get("llm.input_messages.1.message.contents.0.message_content.type") == "text"
    assert attrs.get("llm.input_messages.1.message.contents.1.message_content.type") == "image"
    img_key = "llm.input_messages.1.message.contents.1.message_content.image.image.url"
    assert img_key in attrs, f"Image missing! Keys: {[k for k in attrs if 'image' in k]}"
    assert attrs[img_key] != "__REDACTED__"

    # Assistant with plain string (still works)
    assert attrs.get("llm.input_messages.2.message.content") == (
        "I see tissue with abnormal patterns."
    )

    # Tool with ListConfig [text, image] → mixed → structured only
    assert attrs.get("llm.input_messages.3.message.role") == "tool"
    assert "llm.input_messages.3.message.content" not in attrs
    assert attrs.get("llm.input_messages.3.message.contents.1.message_content.type") == "image"

    # Tool with plain string
    assert attrs.get("llm.input_messages.4.message.content") == (
        "Invalid tool call: missing argument"
    )

    # All 2 images consumed
    image_url_keys = [k for k in attrs if k.endswith(".image.image.url")]
    assert len(image_url_keys) == 2, f"Expected 2 images, got {len(image_url_keys)}"

    provider.shutdown()


async def test_root_span_metadata_from_rollout_trace_op():
    """Verifies that rollout_trace_op sets scalar inputs on the root (rollout_trace_attr) span.

    This mirrors the production flow:
    1. rollout_trace_attr creates the top-level 'agent_loop' span
    2. rollout_trace_op creates a child span for 'run()'
    3. Scalar kwargs (wsi_name, data_source, etc.) should be set on the root span
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openinference.instrumentation import TracerProvider as OITracerProvider

    exporter = InMemorySpanExporter()
    provider = OITracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test-root-metadata")

    rc = RolloutTraceConfig.get_instance()
    rc.backend = "arize"
    rc.client = tracer
    rc._initialized = True

    class FakeResult:
        def __init__(self):
            self.reward_score = 0.85
            self.num_turns = 7
            self.prompt_ids = [1, 2, 3]
            self.response_ids = [4, 5, 6]
            self.response_mask = [1, 1, 1]
            self.response_logprobs = [-0.1, -0.2]
            self.multi_modal_data = {}
            self.extra_fields = {"turn_scores": [0.1, 0.2]}
            self.metrics = {"latency": 1.5}

    class AgentUnderTest:
        @rollout_trace_op
        async def run(self, sampling_params, **kwargs):
            return FakeResult()

    agent = AgentUnderTest()

    with rollout_trace_attr(
        step=5, sample_index=42, rollout_n=3, validate=False, name="agent_loop",
    ):
        await agent.run(
            sampling_params={"temperature": 0.7},
            wsi_name="TCGA-TEST-001",
            data_source="test_dataset",
            primary_site_bb="Breast",
            index=207,
            extra_info={"wsi_id": "TCGA-TEST-001", "slide_level_task": True, "reward_groups": ["a", "b"]},
            messages=[{"role": "system", "content": "hello"}],
            multi_modal_inputs={"image": [1, 2, 3]},
        )

    provider.force_flush()
    exported = exporter.get_finished_spans()
    assert len(exported) == 2, f"Expected 2 spans (child + root), got {len(exported)}"

    # Spans are exported child-first, root-last
    child_attrs = dict(exported[0].attributes)
    root_attrs = dict(exported[1].attributes)

    # Root span should have the original rollout_trace_attr attributes
    assert root_attrs.get("step") == "5"
    assert root_attrs.get("sample_index") == "42"

    # Root span should also have scalar kwargs from rollout_trace_op
    assert root_attrs.get("wsi_name") == "TCGA-TEST-001"
    assert root_attrs.get("data_source") == "test_dataset"
    assert root_attrs.get("primary_site_bb") == "Breast"
    assert root_attrs.get("index") == "207"

    # extra_info scalars should be flattened one level
    assert root_attrs.get("extra_info.wsi_id") == "TCGA-TEST-001"
    assert root_attrs.get("extra_info.slide_level_task") == "True"

    # Non-scalar extra_info values should be skipped
    assert "extra_info.reward_groups" not in root_attrs

    # Skipped keys should not appear on root span
    assert "messages" not in root_attrs
    assert "multi_modal_inputs" not in root_attrs
    assert "sampling_params" not in root_attrs

    # Root span should have scalar result fields (reward_score, num_turns)
    assert root_attrs.get("reward_score") == "0.85"
    assert root_attrs.get("num_turns") == "7"

    # Raw tensor fields from result should NOT appear on root span
    assert "prompt_ids" not in root_attrs
    assert "response_ids" not in root_attrs
    assert "response_mask" not in root_attrs
    assert "response_logprobs" not in root_attrs
    assert "extra_fields" not in root_attrs
    assert "metrics" not in root_attrs

    # Child span should NOT have input/output dumps
    assert "input.value" not in child_attrs
    assert "output.value" not in child_attrs

    provider.shutdown()
