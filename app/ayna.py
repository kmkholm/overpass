# -*- coding: utf-8 -*-
"""Ayna (أين سُجِّل؟) — desktop app: where was this recording made?

Bilingual (Arabic / English) PyQt5 front end over the overpass pipeline. One-file EXE via
Build_EXE.cmd. No torch: the learned detector runs through ONNX Runtime if models/ridgenet.onnx
is present, otherwise the classical scan detector is used.

Tabs:  Verify / Search   (file + date + claimed point or region -> ranked map + report)
       Clean my recording (detect the Doppler comb and remove it before publishing)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_API", "pyqt5")
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
if (BASE / "bin").exists():
    os.environ["PATH"] = str(BASE / "bin") + os.pathsep + os.environ.get("PATH", "")
CACHE = Path(os.environ.get("AYNA_HOME", Path.home() / "Ayna"))
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DOPPLERTRACK_CACHE", str(CACHE / "cache"))

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets as W

APP_VERSION = "0.2.1"
RELIABLE_SCORE = 40.0        # identification-score gate; calibrated on the benchmark (see PROGRESS.md)
DEVELOPER = "Dr. Mohammed Tawfik — kmkhol01@gmail.com"
SETTINGS = CACHE / "settings.json"

# ----------------------------------------------------------------------------- look
ACCENT, ACCENT_D, INK, PAPER, PANEL, LINE, OK, WARN = "#1F6F8B", "#155A72", "#1B2430", "#F4F6F8", "#FFFFFF", "#D9DEE3", "#2E8B57", "#C8102E"
STYLE = f"""
QMainWindow, QWidget {{ background: {PAPER}; color: {INK}; font-size: 10.5pt; }}
QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 8px; background: {PANEL}; top: -1px; }}
QTabBar::tab {{ background: {PAPER}; color: {INK}; padding: 9px 22px; border: 1px solid {LINE}; border-bottom: none;
               border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; font-weight: 600; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {ACCENT}; border-color: {LINE}; }}
QTabBar::tab:hover {{ color: {ACCENT}; }}
QLineEdit, QDateEdit, QSpinBox, QComboBox, QTextEdit {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px; padding: 6px 8px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QDateEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 1.5px solid {ACCENT}; }}
QPushButton {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; padding: 8px 16px; font-weight: 600; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:disabled {{ color: #9AA3AD; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: none; font-size: 12pt; }}
QPushButton#primary:hover {{ background: {ACCENT_D}; }}
QPushButton#primary:disabled {{ background: #9FC3D0; }}
QProgressBar {{ border: 1px solid {LINE}; border-radius: 7px; background: {PANEL}; height: 16px; text-align: center; color: {INK}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QFrame#headerbar QLabel {{ background: transparent; }}
QFrame#headerbar QPushButton {{ background: white; color: {ACCENT_D}; border: none; }}
QLabel#header {{ color: white; font-size: 19pt; font-weight: 700; background: transparent; }}
QLabel#sub {{ color: #D7E6EC; font-size: 10pt; background: transparent; }}
QLabel#intro {{ color: #3E4A56; padding: 4px 2px 8px 2px; }}
QLabel#status {{ color: {ACCENT_D}; font-weight: 600; }}
QFrame#headerbar {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_D}, stop:1 {ACCENT}); border-radius: 10px; }}
QTextEdit#result {{ background: #F8FAFB; font-family: Consolas, 'Courier New'; font-size: 10pt; }}
QStatusBar {{ background: {PAPER}; color: #5B6673; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 8px; margin-top: 10px; padding: 10px 6px 6px 6px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {ACCENT_D}; }}
"""


def _lang() -> str:
    try:
        return json.load(open(SETTINGS)).get("lang", "ar")
    except Exception:
        return "ar"


LANG = _lang()


def L(ar: str, en: str) -> str:
    return ar if LANG == "ar" else en


# ----------------------------------------------------------------------------- pipeline worker
class Worker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str)
    done = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, job: dict):
        super().__init__(); self.job = job

    def run(self):
        try:
            self.done.emit(self._run())
        except RuntimeError as e:           # expected, user-facing conditions
            self.failed.emit(str(e))
        except Exception:
            tb = traceback.format_exc()
            try:
                (CACHE / "ayna_error.log").write_text(tb, encoding="utf-8")
            except Exception:
                pass
            self.failed.emit(L("حدث خطأ غير متوقع. التفاصيل في ", "Unexpected error. Details in ") + str(CACHE / "ayna_error.log") + "\n\n" + tb[-600:])

    # -- source: local file or URL (YouTube or any yt-dlp-supported page, or a direct media link)
    def _source(self, src: str) -> str:
        if not src.lower().startswith(("http://", "https://")):
            return src
        self.progress.emit(2, L("تنزيل الصوت من الرابط…", "Downloading the audio from the link…"))
        dl = CACHE / "downloads"; dl.mkdir(parents=True, exist_ok=True)
        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError(L("تنزيل الروابط غير متاح في هذه النسخة.", "Link download is not available in this build."))
        opts = {"format": "bestaudio/best", "outtmpl": str(dl / "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True,
                "noplaylist": True, "ffmpeg_location": str(BASE / "bin") if (BASE / "bin").exists() else None}
        try:
            with yt_dlp.YoutubeDL({k: v for k, v in opts.items() if v is not None}) as y:
                info = y.extract_info(src, download=True)
                path = y.prepare_filename(info)
        except Exception as e:
            raise RuntimeError(L("تعذّر تنزيل الرابط: ", "Could not download the link: ") + str(e)[:300])
        return path

    # -- shared steps
    def _spectrogram(self, path: str):
        from dopplertrack import pipeline
        from dopplertrack.audio import Spectrogram
        key = Path(path).stem + f"_{Path(path).stat().st_size}"
        sp_path = CACHE / "work" / f"{key}.spec.npz"; sp_path.parent.mkdir(parents=True, exist_ok=True)
        if not sp_path.exists():
            self.progress.emit(5, L("استخراج الصوت وحساب المطياف…", "Decoding audio and computing the spectrogram…"))
            pipeline.make_spectrogram(path, sp_path)
        return Spectrogram.load(sp_path), key

    def _events(self, sp, key: str, top: int):
        from overpass import detect as D
        det_path = CACHE / "work" / f"{key}.det.json"
        if det_path.exists():
            dets = D.load_detections(det_path)
        else:
            onnx = BASE / "models" / "ridgenet.onnx"
            if onnx.exists():
                dur_min = float(sp.t[-1] - sp.t[0]) / 60.0
                self.progress.emit(15, L(f"كشف الطائرات العابرة… (نحو {max(1, dur_min * 0.25):.0f} دقيقة)", f"Detecting flyovers… (about {max(1, dur_min * 0.25):.0f} min)"))
                def prog(f):
                    if f <= 1.0:
                        self.progress.emit(int(15 + 5 * f), L(f"كشف الطائرات العابرة… {f * 100:.0f}%", f"Detecting flyovers… {f * 100:.0f}%"))
                    else:
                        self.progress.emit(int(20 + 8 * (f - 1.0)), L(f"توفيق منحنيات دوبلر… {(f - 1) * 100:.0f}%", f"Fitting Doppler curves… {(f - 1) * 100:.0f}%"))
                dets = D.detect(D.OnnxRidge(onnx), sp, top=top, progress=prog)
            else:
                self.progress.emit(15, L("كشف الطائرات العابرة (المسح الكلاسيكي، أبطأ)…", "Detecting flyovers (classical scan, slower)…"))
                from dopplertrack.tracks import detect_events
                evs = detect_events(sp, 60, 260, top=top, workers=4)
                dets = [D.Detection(e.t_start, e.t_end, e.f_post, e.f_pre, 1, float(e.score) / 20.0, 0, e) for e in evs]
            D.save_detections(dets, det_path)
        return [d.fit for d in dets if d.fit is not None]

    def _traces(self, date: str, lat: float, lon: float):
        from dopplertrack import pipeline
        from dopplertrack.adsb import load_traces
        from overpass.benchmark import bbox_around
        tr_path = CACHE / "work" / f"adsb_{date}_{lat:.2f}_{lon:.2f}.npz"
        if not tr_path.exists():
            self.progress.emit(30, L("جلب مسارات الطيران لذلك اليوم من adsb.lol (قد يستغرق 10–20 دقيقة)…",
                                     "Fetching that day's flight tracks from adsb.lol (10–20 min)…"))
            pipeline.fetch_region(date, bbox_around(lat, lon, 160.0), tr_path)
        return load_traces(tr_path)

    def _run(self) -> dict:
        j = self.job
        if j["kind"] == "scrub":
            return self._scrub()
        from dopplertrack.terrain import Terrain
        from overpass import nsearch as NS
        j["path"] = self._source(j["path"])
        sp, key = self._spectrogram(j["path"])
        dur_min = float(sp.t[-1] - sp.t[0]) / 60.0
        events = self._events(sp, key, top=8)
        self.progress.emit(25, L(f"وُجد {len(events)} حدث طائرة في {dur_min:.0f} دقيقة", f"{len(events)} aircraft events found in {dur_min:.0f} min"))
        if len(events) < 2:
            found = "\n".join(L(f"  حدث عند {e.t_cpa:.0f} ث، أساسي {e.f0:.0f} Hz، سرعة {e.v:.0f} م/ث", f"  event at {e.t_cpa:.0f} s, f0 {e.f0:.0f} Hz, v {e.v:.0f} m/s") for e in events)
            raise RuntimeError(L(
                f"وُجد {len(events)} حدث لطائرة في التسجيل (مدته {dur_min:.0f} دقيقة)، والتحديد يحتاج طائرتين عابرتين على الأقل.\n{found}\n\n"
                "ماذا يساعد: تسجيل أطول (20–60 دقيقة)، في الهواء الطلق أو قرب نافذة، في النهار، في منطقة بها طيران خفيف؛ "
                "والتأكد أن التاريخ المدخل هو تاريخ التسجيل الحقيقي.",
                f"{len(events)} aircraft event(s) found in this {dur_min:.0f}-minute recording; fixing a location needs at least two passing aircraft.\n{found}\n\n"
                "What helps: a longer recording (20–60 min), outdoors or near a window, in daytime, in an area with light aviation; "
                "and make sure the date entered is the real recording date."))
        events = sorted(events[:j.get("n_events", 3)], key=lambda e: e.t_cpa)
        lat0, lon0 = j["center"]
        traces = self._traces(j["date"], lat0, lon0)
        elev = Terrain().elevation
        self.progress.emit(55, L("البحث عن المواقع المرشّحة…", "Searching candidate locations…"))
        if j["kind"] == "verify":
            res, segs = NS.verify(sp, events, traces, lat0, lon0, elevation=elev, half_m=j.get("half_m", 1500.0),
                                  spacing_m=100.0, harmonics=[1] * len(events), progress=False)
        else:
            res, segs = NS.search(sp, events, traces, j["bbox"], elevation=elev, spacing_m=j.get("spacing", 200.0),
                                  harmonics=[1] * len(events), progress=False, cpa_cache=str(CACHE / "work" / f"{key}_cpa.npz"))
        if len(res) == 0:
            raise RuntimeError(L("لا يوجد مرشّح يطابق تسلسل الطائرات المسموعة في هذه المنطقة.",
                                 "No candidate in this region matches the sequence of audible aircraft."))
        import pandas as pd
        best = NS.best_per_point(res)
        tp = NS.top_point(res)
        best = pd.concat([pd.DataFrame([tp]), best[best.gidx != tp.gidx]], ignore_index=True)   # the two-stage answer first
        self.progress.emit(90, L("إعداد الخريطة والتقرير…", "Preparing the map and the report…"))
        out_dir = CACHE / "reports"; out_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"ayna_{stamp}.csv"; best.to_csv(csv_path, index=False)
        png = out_dir / f"ayna_{stamp}.png"; pdf = out_dir / f"ayna_{stamp}.pdf"
        self._figure(best, j, png, pdf, events, sp)
        top = best.iloc[0]
        q = j.get("q90", 1.0)
        from overpass.conformal import gap_set
        cells, area = gap_set(best, q, j.get("spacing", 200.0) if j["kind"] == "search" else 100.0)
        # reliability gate: on recordings with no real flyover the search still returns a best point, but its
        # identification score stays far below that of a genuine match (benchmark: ~9-30 vs 54). Below the gate
        # the answer is reported as "no reliable location" rather than as a pin.
        reliable = float(top.score) >= RELIABLE_SCORE
        return {"lat": float(top.lat), "lon": float(top.lon), "score": float(top.score), "n_events": len(events), "reliable": reliable,
                "n_points": int(len(best)), "area_km2": float(area), "png": str(png), "pdf": str(pdf), "csv": str(csv_path),
                "events": [(float(e.t_cpa), float(e.f0)) for e in events],
                "claimed_rank": self._claimed_rank(best, j) if j["kind"] == "verify" else None}

    @staticmethod
    def _claimed_rank(best, j):
        from dopplertrack.geo import haversine_m
        d = haversine_m(best.lat.to_numpy(), best.lon.to_numpy(), j["center"][0], j["center"][1])
        m = d <= 75.0
        if not m.any():
            return None
        s = float(best.score[m].max())
        return int((best.score > s).sum() + 1)

    def _figure(self, best, j, png, pdf, events, sp):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(best.lon, best.lat, c=best.score, s=4 if len(best) > 2000 else 14, cmap="cividis")
        plt.colorbar(sc, ax=ax, label="score (z)")
        top = best.iloc[0]
        ax.plot(top.lon, top.lat, "+", color="#E63946", ms=14, mew=2, label="best point")
        if j["kind"] == "verify":
            ax.plot(j["center"][1], j["center"][0], "x", color="#00B4D8", ms=12, mew=2, label="claimed point")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect(1 / np.cos(np.radians(top.lat))); ax.legend(frameon=False)
        ax.set_title(f"Ayna — best {top.lat:.5f}, {top.lon:.5f}  (score {top.score:.2f}, {len(events)} events)")
        fig.tight_layout(); fig.savefig(png, dpi=150)
        with PdfPages(pdf) as pp:
            pp.savefig(fig)
            fig2, axes = plt.subplots(len(events), 1, figsize=(7, 2.4 * len(events)), squeeze=False)
            from dopplertrack.tracks import local_whiten
            for a_, e in zip(axes[:, 0], events):
                w = local_whiten(sp.crop(e.t_start - 30, e.t_end + 30, max(e.f_post * 0.7, sp.f[0]), min(e.f_pre * 1.3 * 2, sp.f[-1])))
                a_.imshow(w.S, origin="lower", aspect="auto", extent=[w.t[0], w.t[-1], w.f[0], w.f[-1]], vmin=0, vmax=10, cmap="magma")
                a_.set_title(f"event at {e.t_cpa:.0f} s, f0 {e.f0:.0f} Hz, v {e.v:.0f} m/s, d {e.d:.0f} m", fontsize=9)
            fig2.text(0.01, 0.005, f"Ayna {APP_VERSION} — {DEVELOPER} — file {Path(j['path']).name} — date {j['date']}", fontsize=7)
            fig2.tight_layout(); pp.savefig(fig2); plt.close(fig2)
        plt.close(fig)

    def _scrub(self) -> dict:
        import soundfile as sf
        from dopplertrack.audio import load_audio
        from overpass import scrub as SC
        j = self.job
        j["path"] = self._source(j["path"])
        sp, key = self._spectrogram(j["path"])
        events = self._events(sp, key, top=6)
        self.progress.emit(50, L("إزالة بصمة الطائرات من الصوت…", "Removing the aircraft comb from the audio…"))
        x, sr = load_audio(j["path"], sr=16000)
        tracks = [SC.track_from_event(sp, e, harmonics=3) for e in events]
        y = SC.scrub(x, sr, tracks, mode="fill") if tracks else x
        out = str(Path(j["path"]).with_name(Path(j["path"]).stem + "_clean.wav"))
        sf.write(out, y, sr)
        before_after = [SC.residual_contrast(x, y, sr, e) for e in events]
        return {"out": out, "n_events": len(events), "before_after": before_after,
                "events": [(float(e.t_cpa), float(e.f0)) for e in events]}


# ----------------------------------------------------------------------------- UI
class Main(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(L("أين سُجِّل؟ — Ayna", "Ayna — Where was this recorded?"))
        self.resize(1000, 760)
        if LANG == "ar":
            self.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.setFont(QtGui.QFont("Segoe UI", 10))
        self.setStyleSheet(STYLE)
        icon = BASE / "assets" / "ayna.ico"
        if icon.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon)))
        root = W.QWidget(); self.setCentralWidget(root); v = W.QVBoxLayout(root); v.setContentsMargins(14, 12, 14, 8); v.setSpacing(10)
        # header bar: name, one-line purpose, language toggle
        bar = W.QFrame(); bar.setObjectName("headerbar"); h = W.QHBoxLayout(bar); h.setContentsMargins(18, 12, 18, 12)
        title = W.QLabel(L("أين سُجِّل؟", "Ayna")); title.setObjectName("header")
        sub = W.QLabel(L("تحديد مكان التسجيل من بصمات دوبلر للطائرات العابرة", "Where was this recording made? Aircraft Doppler fingerprints vs public flight tracks")); sub.setObjectName("sub")
        tv = W.QVBoxLayout(); tv.setSpacing(2); tv.addWidget(title); tv.addWidget(sub)
        h.addLayout(tv, 1)
        lang_btn = W.QPushButton(L("English", "العربية")); lang_btn.setFixedWidth(110); lang_btn.clicked.connect(self._switch_lang); h.addWidget(lang_btn)
        v.addWidget(bar)
        tabs = W.QTabWidget(); v.addWidget(tabs, 1)
        tabs.addTab(self._tab_locate(), L("تحقق / بحث", "Verify / Search"))
        tabs.addTab(self._tab_scrub(), L("نظّف تسجيلي", "Clean my recording"))
        tabs.addTab(self._tab_about(), L("حول", "About"))
        self.statusBar().showMessage(f"Ayna {APP_VERSION} — {DEVELOPER}")
        self.thread = None

    # -- helpers
    def _file_row(self, label, attr):
        row = W.QHBoxLayout(); edit = W.QLineEdit(); btn = W.QPushButton(L("اختر ملفاً…", "Browse…"))
        edit.setPlaceholderText(L("ملف صوت/فيديو، أو رابط YouTube / أي رابط فيديو", "Audio/video file, or a YouTube / any video link"))
        def pick():
            p, _ = W.QFileDialog.getOpenFileName(self, label, "", "Audio/Video (*.mp4 *.mkv *.mov *.m4a *.mp3 *.wav *.flac *.ogg *.webm);;All (*)")
            if p: edit.setText(p)
        btn.clicked.connect(pick); row.addWidget(W.QLabel(label)); row.addWidget(edit, 1); row.addWidget(btn)
        setattr(self, attr, edit); return row

    @staticmethod
    def _valid_source(s: str) -> bool:
        return bool(s) and (s.lower().startswith(("http://", "https://")) or Path(s).exists())

    def _tab_locate(self):
        w = W.QWidget(); v = W.QVBoxLayout(w)
        intro = W.QLabel(L("ارفع تسجيلاً صوتياً أو فيديو، حدّد تاريخ التسجيل (UTC)، ثم إمّا نقطة مُدَّعاة للتحقق منها أو منطقة بحث (حتى 150 كم). "
                           "يحتاج البرنامج إلى الإنترنت لجلب مسارات الطيران وبيانات الارتفاع.",
                           "Load an audio or video file, set the recording date (UTC), then either a claimed point to verify or a search region (up to 150 km). "
                           "Internet access is needed for the flight tracks and the elevation data."))
        intro.setWordWrap(True); intro.setObjectName("intro"); v.addWidget(intro)
        v.addLayout(self._file_row(L("الملف:", "File:"), "loc_file"))
        box = W.QGroupBox(L("إعدادات البحث", "Search settings")); g = W.QGridLayout(box); v.addWidget(box)
        self.date = W.QDateEdit(QtCore.QDate.currentDate().addDays(-2)); self.date.setCalendarPopup(True); self.date.setDisplayFormat("yyyy-MM-dd")
        g.addWidget(W.QLabel(L("تاريخ التسجيل (UTC):", "Recording date (UTC):")), 0, 0); g.addWidget(self.date, 0, 1)
        self.mode = W.QComboBox(); self.mode.addItems([L("تحقق من نقطة مُدَّعاة", "Verify a claimed point"), L("بحث في منطقة", "Search a region")])
        g.addWidget(W.QLabel(L("الوضع:", "Mode:")), 0, 2); g.addWidget(self.mode, 0, 3)
        self.lat = W.QLineEdit("40.876466"); self.lon = W.QLineEdit("-73.524951")
        g.addWidget(W.QLabel(L("خط العرض:", "Latitude:")), 1, 0); g.addWidget(self.lat, 1, 1)
        g.addWidget(W.QLabel(L("خط الطول:", "Longitude:")), 1, 2); g.addWidget(self.lon, 1, 3)
        self.side = W.QSpinBox(); self.side.setRange(10, 150); self.side.setValue(100); self.side.setSuffix(" km")
        g.addWidget(W.QLabel(L("ضلع منطقة البحث حول النقطة:", "Search box side around the point:")), 2, 0); g.addWidget(self.side, 2, 1)
        self.nev = W.QSpinBox(); self.nev.setRange(2, 5); self.nev.setValue(3)
        g.addWidget(W.QLabel(L("عدد الطائرات المستخدمة:", "Aircraft events used:")), 2, 2); g.addWidget(self.nev, 2, 3)
        self.run_btn = W.QPushButton(L("ابدأ التحليل", "Start analysis")); self.run_btn.setObjectName("primary"); self.run_btn.setMinimumHeight(42); self.run_btn.clicked.connect(self.start_locate)
        v.addWidget(self.run_btn)
        self.bar = W.QProgressBar(); self.bar.setRange(0, 100); v.addWidget(self.bar)
        self.status = W.QLabel(""); self.status.setWordWrap(True); self.status.setObjectName("status"); v.addWidget(self.status)
        self.result = W.QTextEdit(); self.result.setObjectName("result"); self.result.setReadOnly(True); self.result.setMaximumHeight(150); v.addWidget(self.result)
        self.img = W.QLabel(); self.img.setAlignment(QtCore.Qt.AlignCenter); self.img.setMinimumHeight(260); v.addWidget(self.img, 1)
        row = W.QHBoxLayout()
        self.open_pdf = W.QPushButton(L("افتح التقرير PDF", "Open PDF report")); self.open_pdf.setEnabled(False); self.open_pdf.clicked.connect(lambda: self._open(self._pdf))
        self.open_map = W.QPushButton(L("افتح أفضل نقطة في خرائط Google", "Open best point in Google Maps")); self.open_map.setEnabled(False); self.open_map.clicked.connect(self._gmaps)
        row.addWidget(self.open_pdf); row.addWidget(self.open_map); v.addLayout(row)
        return w

    def _tab_scrub(self):
        w = W.QWidget(); v = W.QVBoxLayout(w)
        intro = W.QLabel(L("يفحص التسجيل بحثاً عن نغمات مراوح الطائرات غير المسموعة التي تكشف موقعك، ويُخرج نسخة منظّفة (WAV) تحتفظ بالكلام. "
                           "لا يحتاج إلى الإنترنت.", "Checks the recording for the inaudible propeller tones that leak your position and writes a cleaned copy (WAV) that keeps the speech. No internet needed."))
        intro.setWordWrap(True); intro.setObjectName("intro"); v.addWidget(intro)
        v.addLayout(self._file_row(L("الملف:", "File:"), "scrub_file"))
        self.scrub_btn = W.QPushButton(L("افحص ونظّف", "Check and clean")); self.scrub_btn.setObjectName("primary"); self.scrub_btn.setMinimumHeight(42); self.scrub_btn.clicked.connect(self.start_scrub); v.addWidget(self.scrub_btn)
        self.bar2 = W.QProgressBar(); self.bar2.setRange(0, 100); v.addWidget(self.bar2)
        self.status2 = W.QLabel(""); self.status2.setWordWrap(True); self.status2.setObjectName("status"); v.addWidget(self.status2)
        self.result2 = W.QTextEdit(); self.result2.setObjectName("result"); self.result2.setReadOnly(True); v.addWidget(self.result2, 1)
        return w

    def _tab_about(self):
        w = W.QWidget(); v = W.QVBoxLayout(w)
        t = W.QTextBrowser(); t.setOpenExternalLinks(True)
        t.setHtml(L(
            f"<h2>أين سُجِّل؟ (Ayna) {APP_VERSION}</h2><p>تحديد مكان تسجيل صوتي أو فيديو من بصمات دوبلر غير المسموعة لمراوح الطائرات العابرة، بمطابقتها مع سجلات الطيران العامة (ADS-B). "
            f"يُستخدم للتحقق الجنائي من مكان تسجيل مُدَّعى، ولحماية خصوصيتك بإزالة هذه البصمة قبل النشر.</p>"
            f"<p><b>المطوّر:</b> {DEVELOPER}</p><p>الكود: <a href='https://github.com/kmkholm/overpass'>github.com/kmkholm/overpass</a> — مبني على dopplertrack (MIT).</p>"
            f"<p>مجلد العمل والتقارير: {CACHE}</p>",
            f"<h2>Ayna {APP_VERSION}</h2><p>Finds where an audio or video recording was made from the inaudible Doppler signatures of passing propeller aircraft, matched against public ADS-B flight tracks. "
            f"For forensic verification of a claimed recording location, and for removing that leak from your own recordings before publishing.</p>"
            f"<p><b>Developer:</b> {DEVELOPER}</p><p>Code: <a href='https://github.com/kmkholm/overpass'>github.com/kmkholm/overpass</a> — built on dopplertrack (MIT).</p>"
            f"<p>Work folder and reports: {CACHE}</p>"))
        v.addWidget(t, 1)
        btn = W.QPushButton(L("English", "العربية")); btn.clicked.connect(self._switch_lang); v.addWidget(btn)
        return w

    def _switch_lang(self):
        json.dump({"lang": "en" if LANG == "ar" else "ar"}, open(SETTINGS, "w"))
        W.QMessageBox.information(self, "Ayna", L("سيُعاد تشغيل البرنامج بالإنجليزية.", "The app will restart in Arabic."))
        os.execl(sys.executable, sys.executable, *sys.argv)

    # -- runs
    def _busy(self, on: bool):
        self.run_btn.setEnabled(not on); self.scrub_btn.setEnabled(not on)

    def start_locate(self):
        p = self.loc_file.text().strip()
        if not self._valid_source(p):
            W.QMessageBox.warning(self, "Ayna", L("اختر ملفاً صالحاً أو الصق رابطاً.", "Choose a valid file or paste a link.")); return
        try:
            lat, lon = float(self.lat.text()), float(self.lon.text())
        except ValueError:
            W.QMessageBox.warning(self, "Ayna", L("الإحداثيات غير صالحة.", "Invalid coordinates.")); return
        date = self.date.date().toString("yyyy-MM-dd")
        if datetime.strptime(date, "%Y-%m-%d").date() >= datetime.now(timezone.utc).date():
            W.QMessageBox.warning(self, "Ayna", L("أرشيف الطيران ليوم ما يتوفر بعد يوم على الأقل. اختر تاريخاً أقدم.",
                                                   "A day's flight archive appears at least one day later. Pick an earlier date.")); return
        kind = "verify" if self.mode.currentIndex() == 0 else "search"
        job = {"kind": kind, "path": p, "date": date, "center": (lat, lon), "n_events": self.nev.value()}
        if kind == "search":
            from overpass.benchmark import bbox_around
            job["bbox"] = bbox_around(lat, lon, float(self.side.value()))
        self.result.clear(); self.img.clear(); self.open_pdf.setEnabled(False); self.open_map.setEnabled(False)
        self._launch(job, self.bar, self.status, self._locate_done)

    def start_scrub(self):
        p = self.scrub_file.text().strip()
        if not self._valid_source(p):
            W.QMessageBox.warning(self, "Ayna", L("اختر ملفاً صالحاً أو الصق رابطاً.", "Choose a valid file or paste a link.")); return
        self.result2.clear()
        self._launch({"kind": "scrub", "path": p}, self.bar2, self.status2, self._scrub_done)

    def _launch(self, job, bar, status, on_done):
        self._busy(True); bar.setValue(1); status.setText(L("بدأ العمل…", "Working…"))
        self.worker = Worker(job); self.thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker.progress.connect(lambda p, m: (bar.setValue(p), status.setText(m)))
        self.worker.done.connect(lambda r: (bar.setValue(100), self._busy(False), on_done(r)))
        self.worker.failed.connect(lambda msg: (self._busy(False), status.setText(L("لم يكتمل.", "Not completed.")),
                                                W.QMessageBox.information(self, "Ayna", msg)))
        self.thread.start()

    def _locate_done(self, r: dict):
        self._pdf = r["pdf"]; self._best = (r["lat"], r["lon"])
        lines = []
        if not r.get("reliable", True):
            lines.append(L(f"⚠ لا يوجد موقع موثوق: أعلى درجة تعرّف {r['score']:.1f} وهي أدنى من عتبة الموثوقية {RELIABLE_SCORE:.0f}. "
                           "غالباً لم تمرّ طائرة مروحية حقيقية قريبة أثناء التسجيل أو أن الصوت ضعيف/مضغوط. النقطة أدناه للعرض فقط.",
                           f"⚠ No reliable location: best identification score {r['score']:.1f} is below the reliability gate {RELIABLE_SCORE:.0f}. "
                           "Most likely no real propeller aircraft passed nearby during the recording, or the audio is weak/compressed. The point below is shown for reference only."))
        lines += [L(f"أفضل نقطة: {r['lat']:.5f}, {r['lon']:.5f} (درجة {r['score']:.2f})", f"Best point: {r['lat']:.5f}, {r['lon']:.5f} (score {r['score']:.2f})"),
                 L(f"عدد الطائرات المستخدمة: {r['n_events']} — نقاط الشبكة المقيّمة: {r['n_points']}", f"Aircraft events used: {r['n_events']} — grid points scored: {r['n_points']}"),
                 L(f"مساحة مجموعة الثقة (درجة ≥ الأفضل − 1): {r['area_km2']:.2f} كم²", f"Confidence set area (score ≥ best − 1): {r['area_km2']:.2f} km²")]
        if r.get("claimed_rank") is not None:
            lines.append(L(f"ترتيب النقطة المُدَّعاة بين نقاط الجوار: {r['claimed_rank']}", f"Rank of the claimed point among its neighbourhood: {r['claimed_rank']}"))
        elif r.get("claimed_rank") is None and "claimed" in str(r):
            pass
        lines.append(L(f"التقرير: {r['pdf']}", f"Report: {r['pdf']}"))
        self.result.setPlainText("\n".join(lines))
        pix = QtGui.QPixmap(r["png"]); self.img.setPixmap(pix.scaled(self.img.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.open_pdf.setEnabled(True); self.open_map.setEnabled(True)
        self.status.setText(L("اكتمل.", "Done."))

    def _scrub_done(self, r: dict):
        lines = [L(f"عدد الطائرات المكتشفة: {r['n_events']}", f"Aircraft events found: {r['n_events']}")]
        for (t, f0), (b, a) in zip(r["events"], r["before_after"]):
            lines.append(L(f"  حدث عند {t:.0f} ث، أساسي {f0:.0f} Hz: تباين البصمة {b:.1f} → {a:.1f} dB", f"  event at {t:.0f} s, f0 {f0:.0f} Hz: comb contrast {b:.1f} → {a:.1f} dB"))
        lines.append(L(f"الملف المنظّف: {r['out']}", f"Cleaned file: {r['out']}"))
        if r["n_events"] == 0:
            lines.append(L("لم تُكتشف بصمة طائرات؛ الملف منسوخ كما هو.", "No aircraft comb was detected; the file was copied unchanged."))
        self.result2.setPlainText("\n".join(lines)); self.status2.setText(L("اكتمل.", "Done."))

    def _open(self, path):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _gmaps(self):
        la, lo = self._best
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(f"https://www.google.com/maps?q={la:.6f},{lo:.6f}"))


def main():
    app = W.QApplication(sys.argv)
    app.setApplicationName("Ayna")
    m = Main(); m.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
