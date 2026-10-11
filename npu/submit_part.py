"""AI Hub profile, laptop half: upload, compile, link and profile one model part on AI Hub.

    B:\\EdgeSeg\\.venv\\Scripts\\python.exe npu/submit_part.py <folder copied down from the machine>

Runs on the laptop, where the AI Hub token is configured. The folder holds what
npu/export_local.py wrote on the rented machine: the serialized source of one model part
and plan.json, the job options Qualcomm's own pipeline would have used. This script submits
the same jobs that pipeline submits (utils/export/compile.py, link.py and profile.py in
qai-hub-models 0.63.0): one compile job per graph of the part, one link job for the part,
one profile job per graph.

Every job id and the profile figures go to results/aihub_profile/ai_hub_jobs_<part>.json, written
after each step, so a failed step keeps the ids of the steps before it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import qai_hub as hub

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "aihub_profile"


def wait(job, record: dict) -> bool:
    """Wait for a job, record how it ended, and say whether it succeeded."""
    status = job.wait()
    record.update(job_id=job.job_id, url=job.url, status=status.code, message=status.message,
                  seconds_waited=round(time.time() - record.pop("_submitted"), 1))
    return bool(status.success)


def main() -> None:
    folder = Path(sys.argv[1])
    plan = json.loads((folder / "plan.json").read_text())
    device = hub.Device(plan["device"])
    part = plan["part"]
    record: dict = {"qai_hub_client": str(hub.__version__), "device": plan["device"], "part": part,
                    "started_unix": time.time(), "uploads": {}, "compile": {}, "link": {}, "profile": {}}

    def save() -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"ai_hub_jobs_{part}.json").write_text(json.dumps(record, indent=1))

    sources: dict[str, hub.Model] = {}
    for graph in plan["graphs"]:
        if graph["source"] not in sources:
            begin = time.time()
            sources[graph["source"]] = hub.upload_model(str(folder / graph["source"]))
            record["uploads"][graph["source"]] = {"model_id": sources[graph["source"]].model_id,
                                                  "seconds": round(time.time() - begin, 1)}
            save()

    compiled = []
    for graph in plan["graphs"]:
        key = f"{graph['component']}||{graph['graph']}"
        entry = record["compile"][key] = {"options": plan["compile_options"].get(key, ""), "_submitted": time.time()}
        job = hub.submit_compile_job(
            model=sources[graph["source"]], device=device, name=f"llama_v3_2_3b_instruct_{part}",
            input_specs={name: (tuple(shape), dtype) for name, (shape, dtype) in graph["input_specs"].items()},
            options=entry["options"])
        ok = wait(job, entry)
        save()
        if not ok:
            raise SystemExit(f"compile job {job.job_id} failed: {entry['message']}")
        compiled.append(job.get_target_model())

    entry = record["link"] = {"options": plan["link_options"], "_submitted": time.time()}
    job = hub.submit_link_job(compiled, device=device, name=f"llama_v3_2_3b_instruct_{part}", options=entry["options"])
    ok = wait(job, entry)
    save()
    if not ok:
        raise SystemExit(f"link job {job.job_id} failed: {entry['message']}")
    linked = job.get_target_model()

    for graph in plan["graphs"]:
        key = f"{graph['component']}||{graph['graph']}"
        entry = record["profile"][key] = {"options": plan["profile_options"].get(key, ""), "_submitted": time.time()}
        job = hub.submit_profile_job(model=linked, device=device, options=entry["options"],
                                     name=f"llama_v3_2_3b_instruct_{part}_{graph['graph']}")
        if wait(job, entry):
            profile = job.download_profile()
            entry["execution_summary"] = profile.get("execution_summary")
        save()

    record["finished_unix"] = time.time()
    save()
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
