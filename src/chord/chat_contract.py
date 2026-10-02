"""Pinned Chat Completions request-field contract shared across boundaries."""

from __future__ import annotations


CHAT_SPEC_PARAMS = frozenset({
    "audio", "frequency_penalty", "function_call", "functions", "logit_bias", "logprobs",
    "max_completion_tokens", "max_tokens", "messages", "metadata", "modalities", "model",
    "moderation", "n", "parallel_tool_calls", "prediction", "presence_penalty",
    "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning_effort",
    "response_format", "safety_identifier", "seed", "service_tier", "stop", "store", "stream",
    "stream_options", "temperature", "tool_choice", "tools", "top_logprobs", "top_p", "user",
    "verbosity", "web_search_options",
})
