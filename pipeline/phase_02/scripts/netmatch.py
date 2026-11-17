"""Point-to-river-network matching utilities (used by 03_river_size.py and 04_match_sites.py; unit-tested
in tests/test_matching.py).

Matching rule (transparent, documented in P2_02):
  1. Candidates: all eligible links within R_MAX (250 m) of the point.
  2. Nearest candidate d1.  An *alternative* is a candidate that does not share a node with the nearest
     link and whose upstream channel length differs by a factor > SIZE_RATIO (2), or carries a different
     non-null watercourse name; it is *competing* if d_alt <= d1 + AMBIG_M (25 m).
  3. Name evidence: a candidate is name-matched if the distinctive word(s) of its watercourse name occur
     in the site label.  If the nearest is not name-matched but a name-matched candidate lies within
     NAME_M (100 m) and d <= d1 + NAME_SLACK_M (50 m), the name-matched candidate is chosen.
  4. Class:  A  nearest link chosen, d <= 50 m, no unresolved competitor (a competitor rejected because
                the nearest link is name-confirmed counts as resolved)
             B  50 < d <= 100 m and no unresolved competitor, or a non-nearest link chosen on name evidence
             C  unresolved competitor with d <= 100 m, or 100 < d <= 250 m
             U  no candidate within 250 m (unmatched; never forced)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import shapely

R_MAX = 250.0
AMBIG_M = 25.0
SIZE_RATIO = 2.0
NAME_M = 100.0
NAME_SLACK_M = 50.0
GENERIC = {"RIVER", "BROOK", "BECK", "BURN", "STREAM", "WATER", "DRAIN", "DYKE", "DIKE", "LODE", "RIVULET",
           "CREEK", "GILL", "GHYLL", "RHYNE", "RHINE", "CUT", "CANAL", "THE", "OF", "AND", "NEW", "OLD",
           "LITTLE", "GREAT", "UPPER", "LOWER", "EAST", "WEST", "NORTH", "SOUTH", "MAIN", "BACK", "MILL", "LEAT",
           "NAVIGATION", "AFON", "NANT", "SEWER", "CARRIER", "TRIBUTARY", "TRIB", "ST"}


def name_tokens(name):
    if not isinstance(name, str) or not name.strip():
        return set()
    toks = set(re.findall(r"[A-Z]+", name.upper().replace("'", "")))
    return {t for t in toks if t not in GENERIC and len(t) >= 3}


def label_tokens(label):
    if not isinstance(label, str):
        return set()
    s = label.upper().replace("'", "")
    s = re.sub(r"\bR\.", " ", s)
    return set(re.findall(r"[A-Z]+", s))


def candidates(px, py, geoms, radius=R_MAX):
    """All (point, link) pairs within radius. Returns DataFrame pt, link, dist, frac (0-1 along link)."""
    pts = shapely.points(np.asarray(px, float), np.asarray(py, float))
    tree = shapely.STRtree(geoms)
    pi, li = tree.query(pts, predicate="dwithin", distance=radius)
    d = shapely.distance(pts[pi], geoms[li])
    frac = shapely.line_locate_point(geoms[li], pts[pi], normalized=True)
    out = pd.DataFrame({"pt": pi, "link": li, "dist": d, "frac": frac})
    return out.sort_values(["pt", "dist", "link"], kind="mergesort").reset_index(drop=True)


def choose(cands_pt, links, label=None):
    """Choose a link for ONE point given its candidate rows (sorted by distance).
    links: DataFrame indexed by link position with columns u, v, up_mid, watercourse_name.
    Returns dict(link, dist, frac, cls, ambiguous, resolved_by_name, name_match, n_cand, alt_link, alt_dist, size_ratio_alt)."""
    if len(cands_pt) == 0:
        return dict(link=-1, dist=np.nan, frac=np.nan, cls="U", ambiguous=False, resolved_by_name=False,
                    name_match=False, n_cand=0, alt_link=-1, alt_dist=np.nan, size_ratio_alt=np.nan)
    c = cands_pt.reset_index(drop=True)
    lt = label_tokens(label)
    nm = np.array([bool(name_tokens(links.at[l, "watercourse_name"]) & lt) for l in c.link])
    first = int(c.link.iloc[0])
    nodes1 = {links.at[first, "u"], links.at[first, "v"]}
    up1 = max(links.at[first, "up_mid"], 1e-3)
    name1 = links.at[first, "watercourse_name"]
    alt_rows = []
    for k in range(1, len(c)):
        l = int(c.link.iloc[k])
        if {links.at[l, "u"], links.at[l, "v"]} & nodes1:
            continue
        upk = max(links.at[l, "up_mid"], 1e-3)
        r = max(upk / up1, up1 / upk)
        nk = links.at[l, "watercourse_name"]
        diff_name = isinstance(nk, str) and isinstance(name1, str) and nk != name1
        if r > SIZE_RATIO or diff_name:
            alt_rows.append((k, r))
    alt_k, alt_r = (alt_rows[0] if alt_rows else (None, np.nan))
    d1 = float(c.dist.iloc[0])
    competing = alt_k is not None and float(c.dist.iloc[alt_k]) <= d1 + AMBIG_M
    chosen = 0
    resolved = False
    ok = np.flatnonzero(nm & (c.dist.values <= max(NAME_M, d1)) & (c.dist.values <= d1 + NAME_SLACK_M))
    if len(ok):
        # several name-matched candidates (e.g. a detached side channel and the main channel of the same river):
        # take the main channel, i.e. the largest upstream channel length among them
        ups = np.array([links.at[int(c.link.iloc[k]), "up_mid"] for k in ok], float)
        best = int(ok[np.nanargmax(np.nan_to_num(ups, nan=-1))])
        if best != 0:
            chosen = best; resolved = True
        elif competing:
            resolved = True      # nearest is name-confirmed: competitor rejected on name evidence
    d = float(c.dist.iloc[chosen])
    unresolved = competing and not resolved
    if unresolved:
        cls = "C"
    elif d <= 50:
        cls = "B" if chosen != 0 else "A"     # re-assigned away from the nearest link by name -> B
    elif d <= 100:
        cls = "B"
    else:
        cls = "C"
    return dict(link=int(c.link.iloc[chosen]), dist=d, frac=float(c.frac.iloc[chosen]), cls=cls,
                ambiguous=bool(competing), resolved_by_name=bool(resolved), name_match=bool(nm[chosen]),
                n_cand=int(len(c)), alt_link=int(c.link.iloc[alt_k]) if alt_k is not None else -1,
                alt_dist=float(c.dist.iloc[alt_k]) if alt_k is not None else np.nan, size_ratio_alt=float(alt_r))


def match_points(px, py, labels, geoms, links, radius=R_MAX):
    """Vectorised candidate search + per-point choice. Returns DataFrame aligned with input order."""
    cand = candidates(px, py, geoms, radius)
    groups = {k: g for k, g in cand.groupby("pt", sort=False)}
    empty = cand.iloc[0:0]
    rows = []
    for i in range(len(px)):
        rows.append(choose(groups.get(i, empty), links, labels[i] if labels is not None else None))
    return pd.DataFrame(rows), cand
