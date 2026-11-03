"""Light tests of the pure-numpy parts (no torch, no network). Run: pytest -q tests"""
import numpy as np
import pandas as pd

from dopplertrack import audio, search, tracks
from dopplertrack.adsb import Trace
from dopplertrack.geo import enu_to_lla
from overpass import conformal as CF
from overpass import contrast as CT
from overpass import nsearch as NS
from overpass import scrub as SC
from overpass import synth


def _trace(icao, t, enu, lat0, lon0):
    lat, lon, alt = enu_to_lla(enu[:, 0], enu[:, 1], enu[:, 2], lat0, lon0)
    return Trace(icao, None, "C172", "synthetic", t, lat, lon, alt, np.full_like(t, 100.0), np.zeros(len(t), bool))


def _three_flights(lat0=40.8765, lon0=-73.525, T0=1_700_000_000.0):
    t1 = T0 + np.arange(0, 900, 5.0)
    enu1 = np.stack([110.0 * (t1 - (T0 + 300)), np.full_like(t1, 2500.0), np.full_like(t1, 1500.0)], -1)
    t2 = T0 + np.arange(900, 1800, 5.0); tau = t2 - (T0 + 1220)
    enu2 = np.stack([3000.0 + 0.02 * tau ** 2, 95.0 * tau, np.full_like(t2, 1200.0)], -1)
    t3 = T0 + np.arange(1800, 2700, 5.0); tau3 = t3 - (T0 + 2300)
    enu3 = np.stack([-80.0 * tau3, -2000.0 + 0.01 * tau3 ** 2, np.full_like(t3, 900.0)], -1)
    traces = [_trace("a1", t1, enu1, lat0, lon0), _trace("a2", t2, enu2, lat0, lon0), _trace("a3", t3, enu3, lat0, lon0)]
    return traces, search.build_segments(traces, lat0, lon0), lat0, lon0, T0


def test_curve_mask_shape_and_peak():
    t = np.arange(0, 120, 0.25); f = 60 + np.arange(256) * (4000 / 8192)
    t_obs = np.linspace(0, 120, 50); ratio = np.linspace(1.1, 0.9, 50)
    M = synth.curve_mask(t, f, t_obs, ratio, 100.0, harmonics=2)
    assert M.shape == (256, 480) and M.max() > 0.9
    # the fundamental at t=60 s is ~100 Hz: the mask column peaks near that bin
    k = int(np.argmax(M[:, 240])); assert abs(f[k] - 100.0) < 1.5


def test_render_tile_and_dataset():
    _, segs, *_ = _three_flights()
    rng = np.random.default_rng(0)
    S, M, n = synth.render_tile(segs, synth.TileSpec(), rng, p_positive=1.0)
    assert S.shape == M.shape == (256, 480) and M.sum() > 0
    X, Y = synth.make_dataset(segs, 4, seed=1)
    assert X.shape == (4, 1, 256, 480) and Y.max() <= 1.0


def test_conformal_quantile_and_sets():
    q = CF.conformal_quantile(np.arange(1, 11, dtype=float), 0.1)
    assert q == 10.0                       # ceil(11*0.9)=10 -> 10th value
    q2 = CF.conformal_quantile(np.arange(1, 101, dtype=float), 0.1)
    assert q2 == 91.0
    rb = pd.DataFrame({"gidx": [0, 1, 2], "lat": [40.0, 40.002, 40.01], "lon": [-73.0, -73.0, -73.0], "score": [10.0, 9.5, 5.0]})
    cells, area = CF.gap_set(rb, 1.0, 200.0)
    assert len(cells) == 2 and abs(area - 2 * 0.04) < 1e-9
    assert CF.covers_gap(cells, 40.002, -73.0, 200.0) and not CF.covers_gap(cells, 40.01, -73.0, 200.0)
    assert CF.gap_nonconformity(rb, 40.002, -73.0, 200.0) == 0.5


def test_chain_candidates_three_events():
    # two grid points; point 0 hears flights A,B,C at offsets 900 and 1800 s; point 1 hears A,B only
    cpa = pd.DataFrame({"gidx": [0, 0, 0, 1, 1], "seg": [0, 1, 2, 0, 1],
                        "t_emit": [0, 900, 1800, 0, 900], "t_heard": [10.0, 912.0, 1805.0, 10.0, 905.0],
                        "d": [3000.0] * 5, "v": [100.0] * 5})
    m = NS.chain_candidates(cpa, [900.0, 1795.0], tol_s=6.0, silence_d_m=None)
    assert list(m.gidx) == [0] and int(m.seg3.iloc[0]) == 2
    m2 = NS.chain_candidates(cpa, [897.0], tol_s=6.0, silence_d_m=None)
    assert set(m2.gidx) == {0, 1}
    # silence: a fourth flight heard at point 0 between events 1 and 2 removes point 0
    cpa2 = pd.concat([cpa, pd.DataFrame({"gidx": [0], "seg": [3], "t_emit": [400], "t_heard": [405.0], "d": [2000.0], "v": [90.0]})], ignore_index=True)
    m3 = NS.chain_candidates(cpa2, [897.0], tol_s=6.0, silence_d_m=6000.0)
    # the (A, B) candidate at point 0 is removed; (B, C) at point 0 (gap 893 s, nothing between) and (A, B) at point 1 stay
    assert not ((m3.gidx == 0) & (m3.seg1 == 0)).any()
    assert ((m3.gidx == 1) & (m3.seg1 == 0) & (m3.seg2 == 1)).any()
    assert ((m3.gidx == 0) & (m3.seg1 == 1) & (m3.seg2 == 2)).any()


def test_contrast_locked_score_prefers_truth():
    traces, segs, lat0, lon0, T0 = _three_flights()
    c = 340.0; sr = 4000
    x = synth.synth_audio(segs, np.zeros(3), [120.0, 130.0, 110.0], T0, 2700, sr=sr, c=c, harmonics=1)
    sp = audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.5, fmin=60, fmax=300, whiten_window_s=None)
    ev1 = tracks.fit_event(sp, 230, 400, 90, 170, c=c); ev2 = tracks.fit_event(sp, 1150, 1310, 100, 180, c=c)
    ev3 = tracks.fit_event(sp, 2200, 2400, 90, 160, c=c)
    wins = [CT.ContrastWindow.from_fit(sp, e) for e in (ev1, ev2, ev3)]
    d = np.linalg.norm(segs[0].enu, axis=1); j = int(np.argmin(d)); t_heard1 = segs[0].t[j] + d[j] / c
    z_true, zs, sh, _ = CT.contrast_locked_score(wins, segs, np.zeros(3), t_heard1, c=c)
    wrong = np.array([4000.0, -3500.0, 0.0])
    dw = np.linalg.norm(segs[0].enu - wrong, axis=1); jw = int(np.argmin(dw)); th_w = segs[0].t[jw] + dw[jw] / c
    z_wrong, *_ = CT.contrast_locked_score(wins, segs, wrong, th_w, c=c)
    assert z_true > z_wrong and z_true > 3.0, (z_true, z_wrong)


def test_scrub_removes_comb():
    traces, segs, lat0, lon0, T0 = _three_flights()
    c = 340.0; sr = 4000
    x = synth.synth_audio(segs[:1], np.zeros(3), [120.0], T0, 700, sr=sr, c=c, harmonics=2, snr_amp=0.08)
    sp = audio.compute_spectrogram(x, sr, nperseg=8192, hop_s=0.25, fmin=40, fmax=450, whiten_window_s=None)
    ev = tracks.fit_event(sp, 230, 400, 90, 170, c=c, harmonics=2)
    tr = SC.track_from_event(sp, ev, harmonics=3)
    y = SC.scrub(x, sr, [tr], mode="fill")
    assert y.shape == x.shape
    b, a = SC.residual_contrast(x, y, sr, ev, harmonics=2, c=c)
    assert b > 3.0 and a < 0.4 * b, (b, a)
    hp = SC.highpass(x, sr, fc=450.0)
    assert np.isfinite(hp).all()


def test_search_mode_rejects_large_region():
    import pytest
    with pytest.raises(ValueError):
        NS.search(None, [], [], (40.0, 42.0, -74.0, -72.0), max_region_km=150.0)
