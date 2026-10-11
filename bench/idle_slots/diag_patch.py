"""Diagnostic patch for llama.cpp at the pinned commit: count WHY a decode step could not reuse
the previous step's graph.

    python bench/idle_slots/diag_patch.py <llama.cpp source dir>

It only adds counters and one log line; it changes no decision. The counters are printed by
llama_perf_context_print (the server calls it after every request), as one line:

    reuse_fail: prev=.. equal_seqs=.. n_tokens=.. n_seq_tokens=.. n_seqs=.. n_seqs_unq=.. mixed=.. kind_or_data=..
                seq_ids=.. n_outputs=.. samplers=.. sampler_outputs=.. cparams_or_arch=.. inputs=.. | first failing input: <type>=<count> ...

Each failed step is counted once, at the first condition that fails, in the order the code
tests them. Counters are for the whole process (not reset per request).
"""

import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(rel: str, pairs: list[tuple[str, str]]) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    if "llm_reuse_fail" in text and rel != "src/llama-graph.h":
        print("already patched:", rel)
        return
    for old, new in pairs:
        assert text.count(old) == 1, (rel, old[:60], text.count(old))
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("patched", rel)


header = (root / "src/llama-graph.h").read_text(encoding="utf-8")
if "llm_reuse_fail" not in header:
    patch("src/llama-graph.h", [
        ("""    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const llm_graph_params & other) const {""",
         """    // specedge diagnostic: why a step could not reuse the previous graph (first failing condition)
    enum { RF_PREV, RF_EQUAL_SEQS, RF_N_TOKENS, RF_N_SEQ_TOKENS, RF_N_SEQS, RF_N_SEQS_UNQ, RF_MIXED, RF_KIND_OR_DATA,
           RF_SEQ_IDS, RF_N_OUTPUTS, RF_SAMPLERS, RF_SAMPLER_OUTPUTS, RF_CPARAMS_OR_ARCH, RF_INPUTS, RF_COUNT };
    static inline long long llm_reuse_fail[RF_COUNT] = {0};
    static inline std::map<std::string, long long> llm_reuse_fail_input;

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const llm_graph_params & other) const {
        const bool specedge_ok = allow_reuse_impl(other);
        return specedge_ok;
    }

    bool allow_reuse_impl(const llm_graph_params & other) const {"""),
        ("""        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {""",
         """        if (!can_reuse_ubatch) {
            if      (ubatch.equal_seqs() != other.ubatch.equal_seqs()) { llm_reuse_fail[RF_EQUAL_SEQS]++; }
            else if (ubatch.n_tokens     != other.ubatch.n_tokens)     { llm_reuse_fail[RF_N_TOKENS]++; }
            else if (ubatch.n_seq_tokens != other.ubatch.n_seq_tokens) { llm_reuse_fail[RF_N_SEQ_TOKENS]++; }
            else if (ubatch.n_seqs       != other.ubatch.n_seqs)       { llm_reuse_fail[RF_N_SEQS]++; }
            else if (ubatch.n_seqs_unq   != other.ubatch.n_seqs_unq)   { llm_reuse_fail[RF_N_SEQS_UNQ]++; }
            else if (ubatch.is_mixed()   != other.ubatch.is_mixed())   { llm_reuse_fail[RF_MIXED]++; }
            else                                                        { llm_reuse_fail[RF_KIND_OR_DATA]++; }
            return false;
        }

        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {"""),
        ("""        if (!can_reuse_ubatch) {
            return false;
        }

        if (n_outputs != other.n_outputs) {
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            return false;
        }
""",
         """        if (!can_reuse_ubatch) {
            llm_reuse_fail[ubatch.data ? RF_SEQ_IDS : RF_KIND_OR_DATA]++;
            return false;
        }

        if (n_outputs != other.n_outputs) {
            llm_reuse_fail[RF_N_OUTPUTS]++;
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            llm_reuse_fail[RF_SAMPLERS]++;
            return false;
        }
"""),
    ])
    text = (root / "src/llama-graph.h").read_text(encoding="utf-8")
    if "#include <map>" not in text:
        text = text.replace("#include <vector>", "#include <vector>\n#include <map>\n#include <string>", 1)
        (root / "src/llama-graph.h").write_text(text, encoding="utf-8")

patch("src/llama-graph.cpp", [
    ("""    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);
""",
     """    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);
        if (!cur && res) {
            llm_graph_params::llm_reuse_fail[llm_graph_params::RF_INPUTS]++;
            llm_graph_params::llm_reuse_fail_input[typeid(*input).name()]++;
        }
"""),
])
text = (root / "src/llama-graph.cpp").read_text(encoding="utf-8")
if "#include <typeinfo>" not in text:
    (root / "src/llama-graph.cpp").write_text("#include <typeinfo>\n" + text, encoding="utf-8")

patch("src/llama-context.cpp", [
    ("""    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {""",
     """    if (!graph_reuse_disable && gf_res_prev_active != res) {
        llm_graph_params::llm_reuse_fail[llm_graph_params::RF_PREV]++;
    }
    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {"""),
    ("""    LLAMA_LOG_INFO("%s:    graphs reused = %10d\\n", __func__, data.n_reused);
""",
     """    LLAMA_LOG_INFO("%s:    graphs reused = %10d\\n", __func__, data.n_reused);
    {
        const long long * f = llm_graph_params::llm_reuse_fail;
        std::string inputs;
        for (const auto & [name, count] : llm_graph_params::llm_reuse_fail_input) {
            inputs += " " + name + "=" + std::to_string(count);
        }
        LLAMA_LOG_INFO("%s: reuse_fail: prev=%lld equal_seqs=%lld n_tokens=%lld n_seq_tokens=%lld n_seqs=%lld n_seqs_unq=%lld mixed=%lld kind_or_data=%lld seq_ids=%lld n_outputs=%lld samplers=%lld sampler_outputs=%lld cparams_or_arch=%lld inputs=%lld | first failing input:%s\\n",
                __func__, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], inputs.c_str());
    }
"""),
])
print("done")
