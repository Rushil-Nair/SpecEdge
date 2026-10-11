"""AI Hub profile, machine half: build one part of Qualcomm's stock Llama 3.2 3B locally.

    HF_HUB_OFFLINE=1 python npu/export_local.py <output dir> [part name, default the first]

Runs on the rented machine with NO AI Hub token and NO Hugging Face token. It does the
heavy step of Qualcomm's export (instantiate the pre-quantized w4a16 model and write the
files AI Hub compiles) and stops before anything is uploaded. It also writes plan.json:
the exact job options Qualcomm's own pipeline would have used, so the same compile, link
and profile jobs can be submitted from the laptop, where the token lives.

Only one model part is written: the check is whether one part can be exported,
compiled and profiled, not the whole model.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import qai_hub as hub
from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.llama_v3_2_3b_instruct import Model
from qai_hub_models.utils.export.compile import to_hub_input_specs

DEVICE_NAME = "Samsung Galaxy S25 (Family)"
RUNTIME = TargetRuntime.GENIEX_QAIRT
PRECISION = Precision.w4a16


def peak_gb() -> float:
    """Peak resident memory of this process so far, in GB (Linux reports kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def attempt(label: str, function):
    """Some option builders may ask AI Hub about the device. Record a failure instead of stopping."""
    try:
        return function()
    except Exception as err:  # noqa: BLE001
        return {"error": f"{label}: {type(err).__name__}: {str(err)[:300]}"}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    plan: dict = {"device": DEVICE_NAME, "runtime": str(RUNTIME), "precision": str(PRECISION), "graphs": []}

    start = time.time()
    model = Model.from_pretrained()                  # Qualcomm's default pre-quantized checkpoint
    plan["instantiate_seconds"] = round(time.time() - start, 1)
    plan["peak_gb_after_instantiate"] = round(peak_gb(), 1)
    print("instantiated", plan["instantiate_seconds"], "s, peak", plan["peak_gb_after_instantiate"], "GB", flush=True)

    specs = model.get_input_spec()
    components = list(model.component_names)
    plan["components"] = components
    plan["all_graphs"] = [list(key) for key in specs.keys()]
    part = sys.argv[2] if len(sys.argv) > 2 else components[0]
    plan["part"] = part
    device = hub.Device(DEVICE_NAME)

    compile_options = attempt("compile options", lambda: {
        f"{c}||{g}": v for (c, g), v in model.get_hub_compile_options(RUNTIME, PRECISION, "", device).items()})
    plan["compile_options"] = compile_options
    plan["link_options"] = attempt("link options", lambda: model.get_component_hub_link_options(part, RUNTIME, ""))
    plan["profile_options"] = attempt("profile options", lambda: {
        f"{c}||{g}": v for (c, g), v in model.get_hub_profile_options(RUNTIME, "").items()})
    plan["shared_source"] = bool(model.get_component_has_shared_source_model(part))

    written: dict[str, str] = {}
    for (component, graph), spec in specs.items():
        if component != part:
            continue
        entry = {"component": component, "graph": graph,
                 "input_specs": {k: [list(v[0]), v[1]] for k, v in to_hub_input_specs(spec).items()}}
        if plan["shared_source"] and written:
            entry["source"] = next(iter(written.values()))       # every graph of this part shares one file
        else:
            begin = time.time()
            path = Path(model.serialize_component_graph(component, graph, out / f"{component}__{graph}", spec))
            entry["serialize_seconds"] = round(time.time() - begin, 1)
            entry["source"] = str(path.relative_to(out))
            files = [p for p in ([path] if path.is_file() else path.rglob("*")) if p.is_file()]
            entry["source_gb"] = round(sum(p.stat().st_size for p in files) / 1e9, 2)
            written[graph] = entry["source"]
            print("serialized", component, graph, entry["serialize_seconds"], "s,", entry["source_gb"], "GB", flush=True)
        plan["graphs"].append(entry)

    plan["peak_gb_total"] = round(peak_gb(), 1)
    plan["total_seconds"] = round(time.time() - start, 1)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print("peak memory", plan["peak_gb_total"], "GB; total", plan["total_seconds"], "s")
    print("EXPORT_LOCAL_DONE")


if __name__ == "__main__":
    main()
