"""Phase 2B: freeze covariate standardisation constants and HSGP domain constants used by the model specification.
Writes specifications/standardisation_constants.json. Reads frozen phase_02 / pilot_01 outputs only."""
import json
from pathlib import Path
import numpy as np, pandas as pd, pyogrio
ROOT = Path(__file__).resolve().parents[1]; P2 = ROOT.parent / "phase_02"; PILOT = ROOT.parent / "pilot_01"
S = pd.read_csv(P2 / "data_processed/sites_matched.csv")
m = S[(S.primary_RSN | S.primary_MS) & (S.cls != "U") & S.la_hat.notna()].merge(pd.read_csv(PILOT / "data/site_covariates.csv"), on="site")
e = np.log1p(m.edm_n_5km)
g = pyogrio.read_dataframe(P2 / "data_processed/frame_links_england.gpkg", columns=["F1_inland_river"])
b = g.total_bounds / 1000
P = pd.read_csv(P2 / "data_processed/site_years_p2.csv"); D = P[(P.cls != "U") & P.log_area.notna() & P.log_edm5.notna()]
both = D.groupby(["network", "site"]).year.nunique().eq(2).groupby("network").mean()
out = {"n_sites_basis": int(len(m)), "log_area_mean": float(m.la_hat.mean()), "log_area_sd": float(m.la_hat.std()),
       "log1p_edm5_mean": float(e.mean()), "log1p_edm5_sd": float(e.std()),
       "frame_bbox_km": [float(x) for x in b], "hsgp_centre_km": [float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)],
       "hsgp_half_range_km": [float((b[2] - b[0]) / 2), float((b[3] - b[1]) / 2)], "hsgp_boundary_factor": 1.25, "rho_min_km": 12.0}
out["hsgp_m"] = [int(np.ceil(1.75 * 1.25 * h / 12.0)) for h in out["hsgp_half_range_km"]]
out["share_sites_both_years"] = {k: float(v) for k, v in both.items()}
out["share_sites_both_years_all"] = float(D.groupby("site").year.nunique().eq(2).mean())
json.dump(out, open(ROOT / "specifications/standardisation_constants.json", "w"), indent=2); print(json.dumps(out, indent=1))
