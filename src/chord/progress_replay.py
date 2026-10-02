"""Counts-only ingress probe; never log message text or infer its authorship."""
from .progress import LINES


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def count_ingress(messages: list[dict]) -> dict:
    """Inspect before normalization; statusHistory is deliberately excluded.

    Exact phrase presence is diagnostic, not proof it was generated progress:
    a user can legitimately quote a phrase. Absence concerns only these fields
    and exact spellings, not arbitrary rewritten text or client-side storage.
    """
    fields = ("content", "reasoning_content", "reasoning", "reasoning_text",
              "reasoning_details", "thinking", "provider_specific_fields.reasoning")
    phrases = LINES["image"]
    counts = {field: dict.fromkeys(phrases, 0) for field in fields}
    for message in messages:
        for field in fields:
            if field == "provider_specific_fields.reasoning":
                psf = message.get("provider_specific_fields")
                value = psf.get("reasoning") if isinstance(psf, dict) else None
            else:
                value = message.get(field)
            for text in _strings(value):
                for stage, phrase in phrases.items():
                    counts[field][stage] += text.count(phrase)
    return {"messages": len(messages), "matches": counts}
