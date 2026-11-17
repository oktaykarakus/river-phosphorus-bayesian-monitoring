"""Phase 4 A4/A5: re-run the Phase 3C-F identifiability (p3f_06) and upper-tail (p3f_07) analyses, unchanged, on the
authoritative combined posterior (tag T1_combined). Outputs are redirected to phase_04 (diagnostics/, tables/)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_lib as P  # noqa: E402

L = P.redirect()
import p3f_06_identifiability as I6  # noqa: E402
import p3f_07_upper_tail as U7  # noqa: E402

assert I6.L.ROOT == P.P4 and U7.L.ROOT == P.P4
which = sys.argv[1] if len(sys.argv) > 1 else "both"
if which in ("both", "ident"):
    I6.main(P.TAG, "3cf")
if which in ("both", "tail"):
    U7.main(P.TAG, "3cf", 200)
