"""Make the diagnostic line of diag_patch2.py print to stderr (the server did not show it as a
llama log line).   python bench/idle_slots/diag_patch3.py <llama.cpp source dir>"""

import sys
from pathlib import Path

path = Path(sys.argv[1]) / "src" / "llama-context.cpp"
text = path.read_text(encoding="utf-8")
old = 'LLAMA_LOG_INFO("specedge_diag: ubatches=%lld tokens=%lld hist_n_seqs_unq=[%s] reuse_fail=[%s]\\n", sd_ubatches, sd_tokens, hist.c_str(), fails.c_str());'
new = 'fprintf(stderr, "specedge_diag: ubatches=%lld tokens=%lld hist_n_seqs_unq=[%s] reuse_fail=[%s]\\n", sd_ubatches, sd_tokens, hist.c_str(), fails.c_str()); fflush(stderr);'
if new in text:
    print("already patched")
else:
    assert text.count(old) == 1
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("patched")
