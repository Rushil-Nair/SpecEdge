"""Forced JSON for tool calls: a grammar built from a task's tool list (parts 3 and 6).

    grammar = forced_json.grammar_for(task["tools"])
    post("/completion", {"prompt": ..., "grammar": grammar})

The reply must be one of two things:
  - one or more calls, each exactly  {"name": "<a listed tool>", "parameters": {...}}  and
    joined by "; " the way Llama 3.2 writes several calls. Inside "parameters" only that
    tool's argument names are allowed and every value has the JSON type the schema gives:
    a number is a bare number, a boolean is true or false, a string is a quoted string;
  - or plain text that does not start with "{" or "<" (a question back to the user, a
    refusal). "<" is excluded because a call may begin with the special token <|python_tag|>,
    and plain text starting with it would let any malformed call through.

So the model still decides whether to call; the grammar only makes a call well-formed. It
does not check that required arguments are present or that a name is used once: the Python
harness's version is deliberately simple. The C++ runtime of part 6 builds its own.

llama.cpp at the pinned commit has no Python schema converter (only the C++ one behind the
server's json_schema option, which cannot express "or plain text"), so the grammar is written
here directly in GBNF.
"""

from __future__ import annotations

import json

COMMON = r'''
text ::= [^{<] ( [^\n] | "\n" )*
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\""
integer ::= "-"? ("0" | [1-9] [0-9]*)
number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean ::= "true" | "false"
value ::= string | number | boolean | "null" | array | object
array ::= "[" ( value (", " value)* )? "]"
object ::= "{" ( string ": " value (", " string ": " value)* )? "}"
'''


def literal(text: str) -> str:
    """A GBNF string literal that matches exactly this text."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def value_rule(spec) -> str:
    if not isinstance(spec, dict):
        return "value"
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        return "(" + " | ".join(literal(json.dumps(option, ensure_ascii=False)) for option in spec["enum"]) + ")"
    kind = spec.get("type")
    return {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean",
            "array": "array", "object": "object"}.get(kind if isinstance(kind, str) else "", "value")


def grammar_for(tools: list[dict]) -> str:
    rules, calls = [], []
    for index, tool in enumerate(tools):
        function = tool["function"]
        properties = (function.get("parameters") or {}).get("properties") or {}
        pairs = [f'{literal(json.dumps(name, ensure_ascii=False) + ": ")} {value_rule(spec)}'
                 for name, spec in properties.items()]
        if pairs:
            rules.append(f"pair{index} ::= " + " | ".join(pairs))
            arguments = f'( pair{index} (", " pair{index})* )?'
        else:
            arguments = '""'
        head = literal('{"name": ' + json.dumps(function["name"], ensure_ascii=False) + ', "parameters": {')
        rules.append(f'call{index} ::= {head} {arguments} "}}}}"')
        calls.append(f"call{index}")
    if not calls:                      # no tool is offered: only plain text is possible
        return "root ::= text" + COMMON
    # Llama 3.2 opens a call with the special token <|python_tag|>, which the server does not
    # show in the reply text. Without allowing it here the model is pushed into plain text.
    lines = ['root ::= "<|python_tag|>"? call ("; " call)* | text', "call ::= " + " | ".join(calls)] + rules
    return "\n".join(lines) + COMMON
