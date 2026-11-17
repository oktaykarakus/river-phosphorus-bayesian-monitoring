"""Editorial pass E1: re-render Fig. 6 with the final (VR_01) correlation-aware MC error. No fitting, scoring or new estimation.

Only the thin MC-inclusive intervals of panel (a) change. They are rebuilt by the unchanged plotting function
`correction/scripts/vc_06_manuscript_assets.py::fig6`, with each group's `mcse_correlated` replaced by the stored
VR_01 value `mcse_diff_corrected` (repair/diagnostics/VR_results.json). The fold differences, clustered SEs, thick
intervals, points, panel (b) and styling are untouched. table_s6() of vc_06 is NOT called (it would overwrite Table S6).
Writes figures/Fig6_predictive_validation.{png,pdf} and repair/editorial/VE_01_fig6_check.json.
"""
import importlib.util
import json
from pathlib import Path

VF = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vc06", VF / "correction/scripts/vc_06_manuscript_assets.py")
vc06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(vc06)

R = json.load(open(VF / "repair/diagnostics/VR_results.json"))
final = {**R["folds"], **R["pooled"]}
check = {}
for g, e in ((g, e) for g, e in vc06.EX.items() if g in final):
    f = final[g]
    assert abs(e["diff"] - f["diff"]) < 1e-9 and abs(e["se_clustered"] - f["se_clustered"]) < 1e-9, g
    old = e["mcse_correlated"]; e["mcse_correlated"] = f["mcse_diff_corrected"]
    lo, hi = vc06.inflated(g)
    check[g] = {"mcse_superseded": old, "mcse_final": f["mcse_diff_corrected"], "lo95_plotted": lo, "hi95_plotted": hi,
                "lo95_VR_results": f["per_site_lo95_total"], "hi95_VR_results": f["per_site_hi95_total"],
                "max_abs_diff_vs_VR": max(abs(lo - f["per_site_lo95_total"]), abs(hi - f["per_site_hi95_total"]))}
assert all(c["max_abs_diff_vs_VR"] < 1e-12 for c in check.values())
vc06.fig6()
json.dump(check, open(VF / "repair/editorial/VE_01_fig6_check.json", "w"), indent=1)
print(json.dumps({g: [round(c["lo95_plotted"], 4), round(c["hi95_plotted"], 4), f'{c["max_abs_diff_vs_VR"]:.1e}'] for g, c in check.items()}))
