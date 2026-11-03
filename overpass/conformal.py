"""Conformal location sets.

A single top-scoring pin is not evidence. Split conformal prediction turns the score map into a
set of grid cells that contains the true location with probability >= 1 - alpha, calibrated on
recordings whose location is known (the US calibration clips) and then applied unchanged to the
external clips (EU, Jordan).

Two nonconformity scores are offered:
  gap    : best_score - score(true cell)          -> set = cells with score >= best - q
  radius : horizontal error of the top point (m)  -> set = disc of radius q around the top point
The gap set is the one reported in the paper; the radius set is the ablation (it ignores the
shape of the score surface).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dopplertrack.geo import haversine_m


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of calibration nonconformity scores."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[k - 1]) if k <= n else float("inf")


def true_cell_score(res_best: pd.DataFrame, lat: float, lon: float, spacing_m: float) -> float:
    """Score of the grid cell containing the truth (best per point table); -inf if no candidate."""
    d = haversine_m(res_best.lat.to_numpy(), res_best.lon.to_numpy(), lat, lon)
    m = d <= spacing_m * 0.75
    return float(res_best.score[m].max()) if m.any() else float("-inf")


def gap_nonconformity(res_best: pd.DataFrame, lat: float, lon: float, spacing_m: float) -> float:
    best = float(res_best.score.max())
    s = true_cell_score(res_best, lat, lon, spacing_m)
    return float("inf") if not np.isfinite(s) else max(best - s, 0.0)


def radius_nonconformity(res_best: pd.DataFrame, lat: float, lon: float) -> float:
    top = res_best.iloc[0]
    return float(haversine_m(top.lat, top.lon, lat, lon))


def gap_set(res_best: pd.DataFrame, q: float, spacing_m: float) -> tuple[pd.DataFrame, float]:
    """Cells within q of the best score; returns (cells, area_km2)."""
    cells = res_best[res_best.score >= res_best.score.max() - q]
    return cells, len(cells) * (spacing_m ** 2) / 1e6


def radius_set(res_best: pd.DataFrame, q: float) -> tuple[tuple[float, float], float]:
    top = res_best.iloc[0]
    return (float(top.lat), float(top.lon)), math.pi * q * q / 1e6


def covers_gap(cells: pd.DataFrame, lat: float, lon: float, spacing_m: float) -> bool:
    if len(cells) == 0:
        return False
    return bool((haversine_m(cells.lat.to_numpy(), cells.lon.to_numpy(), lat, lon) <= spacing_m * 0.75).any())


def calibrate(nonconf: list[float], alphas=(0.1, 0.2)) -> dict:
    return {f"q{int(round(a * 100))}": conformal_quantile(np.array(nonconf), a) for a in alphas}


def evaluate_sets(items: list[dict], q: float, spacing_m: float, kind: str = "gap") -> pd.DataFrame:
    """items: [{clip, res_best, lat, lon}] from the test split. Returns per-clip coverage and area."""
    rows = []
    for it in items:
        rb = it["res_best"]
        if kind == "gap":
            cells, area = gap_set(rb, q, spacing_m)
            cov = covers_gap(cells, it["lat"], it["lon"], spacing_m)
        else:
            (la, lo), area = radius_set(rb, q)
            cov = haversine_m(la, lo, it["lat"], it["lon"]) <= q
        err = radius_nonconformity(rb, it["lat"], it["lon"])
        rows.append({"clip": it["clip"], "covered": bool(cov), "area_km2": area, "top_error_m": err,
                     "n_cells": len(rb), "kind": kind})
    return pd.DataFrame(rows)
