"""The fix under test, for llama.cpp at commit f498f864: llama-server takes the lowest free slot
instead of the least recently used one when every slot has its own KV stream.

    python bench/idle_slots/fix_patch.py <llama.cpp source dir>

Why: with the per-slot KV cache (the default, kv_unified = false) llama_decode can only put
sequences with CONSECUTIVE ids into one forward pass (llama_batch_allocr::split_equal with
sequential = true: "accept only increasing sequence ids"). Least-recently-used selection walks
through all slots, so with fewer requests than slots the busy slots are scattered and every
decode step is split into several forward passes. The lowest free slot keeps them adjacent.
The prompt-similarity selection that runs before this fallback is not touched.
"""

import sys
from pathlib import Path

path = Path(sys.argv[1]) / "tools" / "server" / "server-context.cpp"
text = path.read_text(encoding="utf-8")
old = """                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
"""
new = """                // with one KV stream per slot, llama_decode only puts sequences with consecutive ids into
                // one forward pass (llama_batch_allocr::split_equal, sequential = true). Take the lowest free
                // slot so that the busy slots stay next to each other; least-recently-used scatters them
                // and every decode step is then split into several forward passes.
                if (!params_base.kv_unified) {
                    if (!ret) {
                        t_last = slot.t_last_used;
                        ret = &slot;
                    }
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
"""
if "Take the lowest free" in text:
    print("already patched")
else:
    assert text.count(old) == 1
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", path)
