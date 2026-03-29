"""Conversation context utilities for bounded, long-running agent sessions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from src.runtime.mlx_adapter import _metal_safe_prompt_token_cap


Message = dict[str, str]
PromptFormatter = Callable[[list[Message]], str]


@dataclass
class BudgetResult:
    """Result of enforcing a context budget."""

    messages: list[Message]
    prompt_tokens: int
    action: str


class EpisodicBuffer:
    """Compress older tool interactions into compact summaries."""

    def __init__(self, recent_pairs: int = 3, max_summary_entries: int = 20) -> None:
        self.recent_pairs = max(1, recent_pairs)
        self.max_summary_entries = max(1, max_summary_entries)

    def compress_messages(self, messages: list[Message]) -> list[Message]:
        """Keep the newest tool exchanges verbatim and summarize the rest."""
        if len(messages) <= 8:
            return list(messages)

        system_message = messages[0]
        protected_messages = [message for message in messages[1:] if message.get("protected")]
        body_messages = [message for message in messages[1:] if not message.get("protected")]

        recent_slots = self.recent_pairs * 2
        preserved_tail = body_messages[-recent_slots:]
        older_messages = body_messages[:-recent_slots]

        if not older_messages:
            return list(messages)

        summary_lines: list[str] = []
        tool_step_number = 0
        index_value = 0
        while index_value < len(older_messages):
            current_message = older_messages[index_value]
            role_name = current_message.get("role", "unknown")
            content_text = current_message.get("content", "").replace("\n", " ").strip()

            if role_name == "assistant":
                if index_value + 1 < len(older_messages) and older_messages[index_value + 1].get("role") == "tool":
                    tool_step_number += 1
                    tool_content = older_messages[index_value + 1].get("content", "").replace("\n", " ").strip()
                    assistant_preview = content_text[:80] or "(empty assistant response)"
                    tool_preview = tool_content[:120] or "(empty tool result)"
                    summary_lines.append(
                        f"Step {tool_step_number}: assistant={assistant_preview} | tool={tool_preview}"
                    )
                    index_value += 2
                    continue

                summary_lines.append(f"Assistant: {content_text[:120] or '(empty assistant response)'}")
            elif role_name == "tool":
                tool_step_number += 1
                summary_lines.append(f"Step {tool_step_number}: tool={content_text[:120] or '(empty tool result)'}")
            elif role_name == "user":
                summary_lines.append(f"User: {content_text[:120] or '(empty user message)'}")

            index_value += 1

        summary_lines = summary_lines[-self.max_summary_entries:]
        if not summary_lines:
            return list(messages)

        summary_message = {
            "role": "user",
            "content": "Compressed session summary:\n" + "\n".join(summary_lines),
            "protected": True,
        }
        retained_protected = protected_messages[-4:]
        return [system_message, *retained_protected, summary_message, *preserved_tail]

    def flush_old(self, messages: list[Message]) -> list[Message]:
        """Public alias used by the idle scheduler."""
        return self.compress_messages(messages)


class ContextBudgetGuard:
    """Enforce a hard prompt budget before each model generation."""

    def __init__(
        self,
        tokenizer,
        context_window: int,
        max_tokens: int,
        episodic_buffer: EpisodicBuffer,
    ) -> None:
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.episodic_buffer = episodic_buffer

    def prompt_tokens(self, messages: list[Message], formatter: PromptFormatter) -> int:
        """Count prompt tokens with tokenizer fallback."""
        try:
            formatted_prompt = formatter(messages)
            return len(self.tokenizer.encode(formatted_prompt))
        except Exception:
            return sum(len(message.get("content", "")) for message in messages) // 4

    def _shrink_messages_to_hard_limit(
        self,
        messages: list[Message],
        formatter: PromptFormatter,
        hard_limit: int,
    ) -> tuple[list[Message], int]:
        """Ensure formatted prompt fits hard_limit by truncating long message bodies (system first)."""
        working_messages = copy.deepcopy(messages)
        marker = "\n\n...[truncated for token budget]\n\n"
        min_tail_chars = 400

        def count_tokens() -> int:
            return self.prompt_tokens(working_messages, formatter)

        if count_tokens() <= hard_limit:
            return working_messages, count_tokens()

        for message_index in range(len(working_messages)):
            content_text = working_messages[message_index].get("content", "")
            if not isinstance(content_text, str) or len(content_text) <= min_tail_chars + len(marker):
                continue

            def trial_content_for_prefix_len(prefix_len: int) -> str:
                if prefix_len >= len(content_text):
                    return content_text
                head = content_text[:prefix_len]
                return head + marker + content_text[-min_tail_chars:]

            low_len = 0
            high_len = len(content_text)
            best_prefix = 0
            while low_len <= high_len:
                mid_len = (low_len + high_len) // 2
                working_messages[message_index]["content"] = trial_content_for_prefix_len(mid_len)
                if count_tokens() <= hard_limit:
                    best_prefix = mid_len
                    low_len = mid_len + 1
                else:
                    high_len = mid_len - 1

            working_messages[message_index]["content"] = trial_content_for_prefix_len(best_prefix)
            if count_tokens() <= hard_limit:
                return working_messages, count_tokens()

        if working_messages:
            working_messages[0]["content"] = (
                "(System prompt truncated to emergency stub; shrink frozen static or raise context_window.)\n"
                + str(working_messages[0].get("content", ""))[:800]
            )
        return working_messages, count_tokens()

    def enforce_budget(self, messages: list[Message], formatter: PromptFormatter) -> BudgetResult:
        """Compress the conversation until it fits a safe prompt budget."""
        working_messages = list(messages)
        prompt_token_count = self.prompt_tokens(working_messages, formatter)
        soft_limit = int(self.context_window * 0.75)
        hard_limit = self.context_window - self.max_tokens - 512
        metal_prompt_cap = _metal_safe_prompt_token_cap()
        if metal_prompt_cap is not None:
            hard_limit = min(hard_limit, metal_prompt_cap)

        if prompt_token_count > soft_limit:
            working_messages = self.episodic_buffer.compress_messages(working_messages)
            prompt_token_count = self.prompt_tokens(working_messages, formatter)
            if prompt_token_count <= hard_limit:
                return BudgetResult(working_messages, prompt_token_count, "episodic_flush")

        action = "none"
        if prompt_token_count > hard_limit:
            protected_messages = [message for message in working_messages[1:] if message.get("protected")]
            minimal_messages = [working_messages[0], *protected_messages[-3:], *working_messages[-3:]]
            working_messages = minimal_messages
            prompt_token_count = self.prompt_tokens(working_messages, formatter)
            action = "minimal_fallback"

        if prompt_token_count > hard_limit:
            working_messages, prompt_token_count = self._shrink_messages_to_hard_limit(
                working_messages, formatter, hard_limit
            )
            action = f"{action}+hard_shrink" if action != "none" else "hard_shrink"

        return BudgetResult(working_messages, prompt_token_count, action)


class KVCacheManager:
    """Reset prompt cache only when the stable prefix changes."""

    def __init__(self, cache_factory: Callable[[], object] | None) -> None:
        self._cache_factory = cache_factory
        self._prompt_cache = cache_factory() if cache_factory else None
        self._last_prefix = ""

    @property
    def prompt_cache(self):
        """Expose the managed prompt cache object."""
        return self._prompt_cache

    def invalidate(self) -> None:
        """Reset the cache to an empty state."""
        if not self._cache_factory:
            self._prompt_cache = None
            self._last_prefix = ""
            return
        self._prompt_cache = self._cache_factory()
        self._last_prefix = ""

    def ensure_prefix(self, prefix_text: str) -> bool:
        """Ensure the cache matches the current stable prompt prefix."""
        if prefix_text == self._last_prefix:
            return False
        self.invalidate()
        self._last_prefix = prefix_text
        return True
