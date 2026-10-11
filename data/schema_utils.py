"""Schema helpers shared by the test-set builder: xLAM parameters to JSON schema, and
When2Call type names to JSON schema type names."""
from __future__ import annotations

JSON_TYPES = {
    "str": "string", "string": "string", "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "dict": "object", "object": "object",
}


def to_json_schema(parameters: dict, keep_defaults: bool = False) -> dict:
    """Convert xLAM's {name: {description, type, default}} into a JSON schema object."""
    properties, required = {}, []
    for name, spec in parameters.items():
        raw_type = str(spec.get("type", "str"))
        base = raw_type.split(",")[0].strip().lower().split("[")[0]
        properties[name] = {
            "type": JSON_TYPES.get(base, "string"),
            "description": str(spec.get("description", "")),
        }
        # Keep the tool's stated default: the model should see it, and the scorer accepts an
        # extra argument only when it restates the default. (Dropped until toolset v0b.)
        if keep_defaults and spec.get("default") not in (None, ""):
            properties[name]["default"] = spec["default"]
        # A parameter with no default and not marked optional must be supplied.
        if "optional" not in raw_type.lower() and spec.get("default") in (None, ""):
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


JSON_TYPE_NAMES = {"float": "number", "dict": "object", "tuple": "array"}


def json_type_names(schema):
    """Replace When2Call's Python type names with JSON schema ones, at every depth."""
    if isinstance(schema, list):
        return [json_type_names(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    return {key: JSON_TYPE_NAMES.get(value, value) if key == "type" and isinstance(value, str) else json_type_names(value)
            for key, value in schema.items()}
