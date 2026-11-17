"""T1 step 4b: apply the unchanged Phase 3C-F gate / estimand code (phase_03c_f/scripts/p3f_04_main_diagnostics.py, imported
read-only) to (a) the combined 4 x 5,000 draws [decision basis] and (b) the T1 segment alone [reported].
Only the module's TAG and the output root (-> addendum_T1) are changed; gates come from phase3cf_settings.yaml main_gates.
Usage: t1_04_gates.py combined|segment
Outputs (addendum only): diagnostics/<TAG>_diagnostics.json, tables/<TAG>_delta_T.csv, tables/<TAG>_ppc_*.csv; scratch <TAG>_delta_T_draws.npz
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_lib as T  # noqa: E402

T.redirect()
import p3f_04_main_diagnostics as M  # noqa: E402

M.TAG = {"combined": T.COMB_TAG, "segment": T.SEG_TAG}[sys.argv[1]]
assert M.L.ROOT == T.ADD
M.main()
