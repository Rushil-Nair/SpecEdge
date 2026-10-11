"""Second diagnostic patch (applied after diag_patch.py): how many forward passes does one
llama_decode call make, and how many sequences does each carry?

    python bench/idle_slots/diag_patch2.py <llama.cpp source dir>

Counters only. Every 2,000 forward passes (ubatches) llama_context::process_ubatch logs one line:

    specedge_diag: ubatches=N tokens=T hist_n_seqs_unq=[c1,c2,...,c16] reuse_fail=[...]

hist_n_seqs_unq[k-1] is the number of forward passes that carried k different sequences.
With plain decoding and U requests in flight, one decode step writes one token per request:
if every step were one forward pass, all passes would carry U sequences.
"""

import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "src" / "llama-context.cpp"
text = path.read_text(encoding="utf-8")
if "specedge_diag" in text:
    print("already patched")
    raise SystemExit(0)
old = """    if (!graph_reuse_disable && gf_res_prev_active != res) {
        llm_graph_params::llm_reuse_fail[llm_graph_params::RF_PREV]++;
    }
"""
new = """    {
        static long long sd_ubatches = 0, sd_tokens = 0, sd_hist[17] = {0};
        sd_ubatches++;
        sd_tokens += ubatch.n_tokens;
        sd_hist[std::min<uint32_t>(ubatch.n_seqs_unq, 16)]++;
        if (sd_ubatches % 2000 == 0) {
            std::string hist, fails;
            for (int i = 1; i <= 16; ++i) { hist += (i > 1 ? "," : "") + std::to_string(sd_hist[i]); }
            for (int i = 0; i < llm_graph_params::RF_COUNT; ++i) { fails += (i > 0 ? "," : "") + std::to_string(llm_graph_params::llm_reuse_fail[i]); }
            LLAMA_LOG_INFO("specedge_diag: ubatches=%lld tokens=%lld hist_n_seqs_unq=[%s] reuse_fail=[%s]\\n", sd_ubatches, sd_tokens, hist.c_str(), fails.c_str());
        }
    }
    if (!graph_reuse_disable && gf_res_prev_active != res) {
        llm_graph_params::llm_reuse_fail[llm_graph_params::RF_PREV]++;
    }
"""
assert text.count(old) == 1
path.write_text(text.replace(old, new), encoding="utf-8")
print("patched", path)
