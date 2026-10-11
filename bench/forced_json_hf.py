"""Forced JSON for generation through Hugging Face-style generate() (part 12, simulated w4a16).

llama.cpp applies bench/forced_json.py's grammar itself. Qualcomm's simulated model generates
through a Hugging Face generator, which knows nothing of grammars, so the same rule is
enforced here as a logits processor:

  the reply is one or more calls  {"name": "<a listed tool>", "parameters": {...}}  joined by
  "; ", with only that tool's argument names and values of the JSON type the schema gives,
  or plain text that does not start with "{".

Decoding is greedy, so at each step only one token is needed: candidates are tried from the
most likely down, and the first whose text keeps the reply a valid beginning is taken. The
rule is written as one regular expression per task and tested with partial matching (the
`regex` package). An end-of-turn token is allowed only when the reply is complete, and the
call-opening special token <|python_tag|> only as the very first token.

Same simplifications as the llama.cpp grammar: required arguments and repeated names are not
checked.
"""

from __future__ import annotations

import json

import regex
import torch
from transformers import LogitsProcessor

DEFINE = (r'(?(DEFINE)'
          r'(?<str>"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*")'
          r'(?<int>-?(?:0|[1-9][0-9]*))'
          r'(?<num>(?&int)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)'
          r'(?<val>(?&str)|(?&num)|true|false|null|(?&arr)|(?&obj))'
          r'(?<arr>\[(?:(?&val)(?:, (?&val))*)?\])'
          r'(?<obj>\{(?:(?&str): (?&val)(?:, (?&str): (?&val))*)?\})'
          r')')


def value_pattern(spec) -> str:
    if not isinstance(spec, dict):
        return "(?&val)"
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        return "(?:" + "|".join(regex.escape(json.dumps(option, ensure_ascii=False)) for option in spec["enum"]) + ")"
    kind = spec.get("type")
    return {"string": "(?&str)", "integer": "(?&int)", "number": "(?&num)", "boolean": "(?:true|false)",
            "array": "(?&arr)", "object": "(?&obj)"}.get(kind if isinstance(kind, str) else "", "(?&val)")


def pattern_for(tools: list[dict]) -> regex.Pattern:
    calls = []
    for tool in tools:
        function = tool["function"]
        properties = (function.get("parameters") or {}).get("properties") or {}
        pairs = [regex.escape(json.dumps(name, ensure_ascii=False) + ": ") + value_pattern(spec)
                 for name, spec in properties.items()]
        arguments = f"(?:(?:{'|'.join(pairs)})(?:, (?:{'|'.join(pairs)}))*)?" if pairs else ""
        calls.append(regex.escape('{"name": ' + json.dumps(function["name"], ensure_ascii=False) + ', "parameters": {')
                     + arguments + regex.escape("}}"))
    text = r"[^{][\s\S]*"
    if not calls:
        return regex.compile(DEFINE + text)
    call = "(?:" + "|".join(calls) + ")"
    return regex.compile(DEFINE + f"(?:{call}(?:; {call})*|{text})")


class ForcedJson(LogitsProcessor):
    """Greedy constrained decoding for one prompt at a time (batch size 1)."""

    def __init__(self, tokenizer, tools: list[dict], prompt_length: int, end_token_ids: list[int], tries: int = 200):
        self.tokenizer, self.pattern, self.prompt_length = tokenizer, pattern_for(tools), prompt_length
        self.ends, self.tries = set(end_token_ids), tries
        self.python_tag = tokenizer.convert_tokens_to_ids("<|python_tag|>")
        self.special = set(tokenizer.all_special_ids)
        self.steps_changed = 0          # steps where the model's first choice was not allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[0, self.prompt_length:].tolist()
        body = [t for t in generated if t != self.python_tag]
        so_far = self.tokenizer.decode(body, skip_special_tokens=True)
        complete = bool(so_far) and self.pattern.fullmatch(so_far) is not None
        order = torch.argsort(scores[0], descending=True)[:self.tries].tolist()
        chosen = None
        for rank, token in enumerate(order):
            if token in self.ends:
                allowed = complete
            elif token == self.python_tag:
                allowed = not generated
            elif token in self.special:
                allowed = False
            else:
                text = self.tokenizer.decode(body + [token], skip_special_tokens=True)
                allowed = len(text) > len(so_far) and self.pattern.fullmatch(text, partial=True) is not None
            if allowed:
                chosen, self.steps_changed = token, self.steps_changed + (rank > 0)
                break
        if chosen is None:              # nothing among the likeliest tokens fits: let the reply end
            chosen = next(iter(self.ends))
        forced = torch.full_like(scores, float("-inf"))
        forced[0, chosen] = 0.0
        return forced
