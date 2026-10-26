#!/usr/bin/env python3
"""Renders the `chaosmap lseis-study` outputs of a case with PyCairo:

- `lseis_heatmap.png`: station x time heatmap of the Wolf exponent's excess
  over its IAAFT surrogates, between the catalog aftershock rate and the daily
  share of windows outside the surrogate range.
- `lseis_station_map.png`: fault map with one marker per station for the raw
  post-minus-pre Wolf change and for the excess over the surrogates. Stations
  are not interpolated: four stations cannot support a field.
- `lseis_timelapse.mp4` (with `--video`, needs ffmpeg): the station map and
  the heatmap advancing one 3 h window per frame, with catalog events.

    python3 tools/plot_lseis.py configs/van2011.toml --tag long_mc --video
"""

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import shutil
import subprocess
import tempfile
import tomllib
from collections import defaultdict

import cairo

# Chart tokens (light surface)
SURFACE = (0xFC / 255, 0xFC / 255, 0xFB / 255)
PAGE = (0xF9 / 255, 0xF9 / 255, 0xF7 / 255)
INK = (0x0B / 255, 0x0B / 255, 0x0B / 255)
INK_2 = (0x52 / 255, 0x51 / 255, 0x4E / 255)
MUTED = (0x89 / 255, 0x87 / 255, 0x81 / 255)
GRID = (0xE1 / 255, 0xE0 / 255, 0xD9 / 255)
AXIS = (0xC3 / 255, 0xC2 / 255, 0xB7 / 255)
FONT = "Sans"
SCALE = 2  # device pixels per layout unit

STRIDE_DAYS = 600.0 / 86400.0 * 18  # study window grid: one window per 3 h


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# --- Diverging scale: blue (negative) <-> gray <-> red (positive), in OKLab --

def _lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gam(c):
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def to_oklab(rgb):
    r, g, b = (_lin(c) for c in rgb)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def from_oklab(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (_gam(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
            _gam(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
            _gam(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s))


NEUTRAL = hex_rgb("#f0efec")
NEG_ARM = [to_oklab(hex_rgb(h)) for h in ("#f0efec", "#86b6ef", "#2a78d6", "#184f95")]
POS_ARM = [to_oklab(hex_rgb(h)) for h in ("#f0efec", "#f2a19c", "#e34948", "#a32626")]


def diverging(v, lim):
    """Color of `v` on a symmetric scale [-lim, lim]."""
    t = max(-1.0, min(1.0, v / lim))
    arm = NEG_ARM if t < 0 else POS_ARM
    x = abs(t) * (len(arm) - 1)
    i = min(len(arm) - 2, int(x))
    f = x - i
    a, b = arm[i], arm[i + 1]
    return from_oklab(tuple(a[k] + f * (b[k] - a[k]) for k in range(3)))


# --- Data ------------------------------------------------------------------

def load_case(config_path, tag):
    config_path = pathlib.Path(config_path).resolve()
    cfg = tomllib.loads(config_path.read_text())
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    def resolve(p):
        p = pathlib.Path(p)
        return p if p.is_absolute() else root / p

    ev, reg, paths = cfg["event"], cfg["region"], cfg["paths"]
    out_dir = resolve(paths["out_dir"])
    sfx = f"_{tag}" if tag else ""
    origin_s = ev["origin_epoch_s"]

    with open(out_dir / f"lseis_study_stations{sfx}.csv") as f:
        stations = list(csv.DictReader(f))
    stations.sort(key=lambda s: float(s["dist_km"]))
    with open(out_dir / f"lseis_study_windows{sfx}.csv") as f:
        windows = list(csv.DictReader(f))
    for w in windows:
        w["rel"] = (float(w["start_mjd"]) - 40587.0) * 86400.0 - origin_s
        w["rel"] /= 86400.0
        w["k"] = round(w["rel"] / STRIDE_DAYS)
        w["excess"] = float(w["wolf"]) - float(w["surr_mean"])
        w["outside"] = float(w["surr_rank"]) in (0.0, 1.0)
    rel = [w["rel"] for w in windows]
    t0, t1 = math.floor(min(rel)), math.ceil(max(rel))

    events = []
    with open(resolve(paths["catalog_csv"]), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                lat, lon = float(r["Latitude"]), float(r["Longitude"])
                if not (reg["lat_min"] <= lat <= reg["lat_max"] and reg["lon_min"] <= lon <= reg["lon_max"]):
                    continue
                t = dt.datetime.strptime(r["Date"].strip(), "%d/%m/%Y %H:%M:%S")
            except (ValueError, KeyError):
                continue
            d = (t.replace(tzinfo=dt.timezone.utc).timestamp() - origin_s) / 86400.0
            if t0 <= d <= t1:
                events.append({"rel": d, "lat": lat, "lon": lon, "mag": float(r["Magnitude"])})

    trace = None
    gj = out_dir / "fault_features.geojson"
    if gj.exists():
        for feat in json.loads(gj.read_text())["features"]:
            if feat["properties"].get("role") == "coseismic_fault_trace":
                trace = feat["geometry"]["coordinates"]

    return {
        "name": ev.get("name", ev.get("id", "case")), "origin_s": origin_s,
        "epi": (ev["lat"], ev["lon"]), "mw": ev.get("mw_ref"),
        "out_dir": out_dir, "sfx": sfx, "stations": stations, "windows": windows,
        "events": events, "trace": trace, "t0": t0, "t1": t1,
    }


def daily_rate(events, t0, t1, min_mag=2.5, until=None):
    counts = defaultdict(int)
    for e in events:
        if e["mag"] >= min_mag and (until is None or e["rel"] <= until):
            counts[math.floor(e["rel"])] += 1
    return {d: counts[d] for d in range(t0, t1)}


def daily_outside(windows, t0, t1, until=None):
    n, k = defaultdict(int), defaultdict(int)
    for w in windows:
        if until is not None and w["rel"] > until:
            continue
        d = math.floor(w["rel"])
        n[d] += 1
        k[d] += w["outside"]
    return {d: (k[d] / n[d] if n[d] else None, n[d]) for d in range(t0, t1)}


def excess_limit(windows):
    """Symmetric color limit: the 99th percentile of |excess|, rounded up."""
    a = sorted(abs(w["excess"]) for w in windows)
    p = a[min(len(a) - 1, int(0.99 * len(a)))] if a else 0.1
    return math.ceil(p * 20) / 20  # to 0.05


def signed(v, nd=2):
    """`v` with an explicit sign, a true minus, and 2..`nd` decimals."""
    if abs(v) < 0.5 * 10 ** -nd:  # no "-0.00"
        v = 0.0
    s = f"{v:+.{nd}f}"
    while s.endswith("0") and len(s.split(".")[1]) > 2:
        s = s[:-1]
    return s.replace("-", "\u2212")


def fmt_date(case, rel_days):
    t = dt.datetime.fromtimestamp(case["origin_s"] + rel_days * 86400.0, dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M UTC")


# --- Drawing helpers --------------------------------------------------------

def text(ctx, x, y, s, size=11, color=INK, bold=False, align="left", valign="baseline"):
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ext = ctx.text_extents(s)
    if align == "center":
        x -= ext.x_advance / 2
    elif align == "right":
        x -= ext.x_advance
    if valign == "middle":
        y += size * 0.35
    elif valign == "top":
        y += size * 0.8
    ctx.set_source_rgb(*color)
    ctx.move_to(x, y)
    ctx.show_text(s)
    return ext.x_advance


def hline(ctx, x0, x1, y, color=GRID, width=1.0):
    ctx.set_source_rgb(*color)
    ctx.set_line_width(width)
    ctx.move_to(x0, y)
    ctx.line_to(x1, y)
    ctx.stroke()


def vline(ctx, x, y0, y1, color=GRID, width=1.0):
    ctx.set_source_rgb(*color)
    ctx.set_line_width(width)
    ctx.move_to(x, y0)
    ctx.line_to(x, y1)
    ctx.stroke()


def nice_ticks(vmax, n=4):
    step = 10 ** math.floor(math.log10(max(vmax, 1e-9) / n))
    for m in (1, 2, 2.5, 5, 10):
        if vmax / (m * step) <= n:
            step *= m
            break
    return [i * step for i in range(math.ceil(vmax / step - 1e-9) + 1)]


def new_surface(w, h):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(w * SCALE), int(h * SCALE))
    ctx = cairo.Context(surf)
    ctx.scale(SCALE, SCALE)
    ctx.set_source_rgb(*SURFACE)
    ctx.paint()
    return surf, ctx


def colorbar_h(ctx, x, y, w, h, lim, label):
    n = 120
    for i in range(n):
        v = -lim + 2 * lim * (i + 0.5) / n
        ctx.set_source_rgb(*diverging(v, lim))
        ctx.rectangle(x + w * i / n, y, w / n + 0.6, h)
        ctx.fill()
    for v in (-lim, -lim / 2, 0.0, lim / 2, lim):
        tx = x + w * (v + lim) / (2 * lim)
        vline(ctx, tx, y + h, y + h + 3, AXIS)
        text(ctx, tx, y + h + 14, signed(v, 3) if v else "0", 10, INK_2, align="center")
    text(ctx, x, y - 6, label, 10.5, INK_2)


# --- Time panels (shared by the heatmap figure and the video) ----------------

def draw_time_panels(ctx, case, box, lim, t_cursor=None, compact=False):
    """Catalog rate, station x time heatmap and daily outside-surrogate share,
    stacked in `box` = (x, y, w, h). With `t_cursor`, data after it is hidden."""
    x, y, w, h = box
    t0, t1 = case["t0"], case["t1"]
    sx = lambda t: x + w * (t - t0) / (t1 - t0)
    stations = case["stations"]
    fs = 9.5 if compact else 10.5
    gap = 14 if compact else 20
    h_rate = h * 0.22
    h_heat = h * 0.42
    h_frac = h - h_rate - h_heat - 2 * gap
    y_rate, y_heat = y, y + h_rate + gap
    y_frac = y_heat + h_heat + gap
    show = (lambda t: True) if t_cursor is None else (lambda t: t <= t_cursor)

    # (a) catalog events per day
    # axis ranges from the full series, so they stay fixed while a video plays
    rmax = max(daily_rate(case["events"], t0, t1).values() or [1])
    rate = daily_rate(case["events"], t0, t1, until=t_cursor)
    ticks = nice_ticks(rmax, 3)
    top = ticks[-1]
    for tv in ticks:
        ty = y_rate + h_rate * (1 - tv / top)
        hline(ctx, x, x + w, ty, GRID if tv else AXIS)
        text(ctx, x - 6, ty, f"{tv:g}", fs - 1, MUTED, align="right", valign="middle")
    bw = w / (t1 - t0)
    for d, c in rate.items():
        if c == 0 or not show(d):
            continue
        bh = h_rate * c / top
        ctx.set_source_rgb(*INK_2)
        ctx.rectangle(sx(d) + 0.3, y_rate + h_rate - bh, max(0.6, bw - 0.6), bh)
        ctx.fill()
    text(ctx, x, y_rate - 6, "Earthquakes per day (M 2.5+)" if compact else
         "Catalogued earthquakes per day (M 2.5+, study region)", fs, INK_2)

    # (b) station x time heatmap of the excess
    row_h = (h_heat - 2 * (len(stations) - 1)) / max(1, len(stations))
    idx = {s["station"]: i for i, s in enumerate(stations)}
    for i, s in enumerate(stations):
        ry = y_heat + i * (row_h + 2)
        ctx.set_source_rgb(*PAGE)
        ctx.rectangle(x, ry, w, row_h)
        ctx.fill()
        text(ctx, x - 6, ry + row_h / 2, s["station"], fs, INK, align="right", valign="middle")
        if not compact:
            text(ctx, x - 6, ry + row_h / 2 + fs + 1, f"{float(s['dist_km']):.0f} km", fs - 1.5,
                 MUTED, align="right", valign="middle")
    cw = w * STRIDE_DAYS / (t1 - t0)
    for wnd in case["windows"]:
        if not show(wnd["rel"]):
            continue
        i = idx.get(wnd["station"])
        if i is None:
            continue
        ry = y_heat + i * (row_h + 2)
        ctx.set_source_rgb(*diverging(wnd["excess"], lim))
        ctx.rectangle(sx(wnd["rel"]), ry, max(0.8, cw), row_h)
        ctx.fill()
    text(ctx, x, y_heat - 6, "Wolf minus surrogate mean, per window" if compact else
         "Wolf exponent minus its IAAFT-surrogate mean, per 10-min window (gray: masked or no data)",
         fs, INK_2)

    # (c) daily share of windows outside the surrogate range, all stations
    frac = daily_outside(case["windows"], t0, t1)
    fmax = max([v for v, n in frac.values() if v is not None] + [0.25])
    if t_cursor is not None:
        frac = daily_outside(case["windows"], t0, t1, until=t_cursor)
    top_f = nice_ticks(fmax, 3)[-1]
    for tv in nice_ticks(top_f, 3):
        ty = y_frac + h_frac * (1 - tv / top_f)
        hline(ctx, x, x + w, ty, GRID if tv else AXIS)
        text(ctx, x - 6, ty, f"{tv * 100:.0f}%", fs - 1, MUTED, align="right", valign="middle")
    ref_y = y_frac + h_frac * (1 - 0.05 / top_f)
    ctx.set_dash([4, 3])
    hline(ctx, x, x + w, ref_y, MUTED)
    ctx.set_dash([])
    ctx.set_source_rgb(*INK)
    ctx.set_line_width(1.6)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    pen = False
    for d in range(t0, t1):
        v, n = frac[d]
        if v is None or n < 4 or not show(d):
            pen = False
            continue
        if d == 0:
            pen = False
        px, py = sx(d + 0.5), y_frac + h_frac * (1 - v / top_f)
        if pen:
            ctx.line_to(px, py)
        else:
            ctx.move_to(px, py)
            pen = True
    ctx.stroke()
    text(ctx, x, y_frac - 6, "Share outside surrogates per day (dashed: 5% chance)" if compact else
         "Share of windows outside their surrogates' range, per day, all stations "
         "(dashed: 5% expected by chance)", fs, INK_2)

    # shared x-axis and mainshock line
    y_axis = y_frac + h_frac
    step = 10 if (t1 - t0) > 40 else 2
    for d in range((t0 // step) * step, t1 + 1, step):
        if d < t0:
            continue
        vline(ctx, sx(d), y_axis, y_axis + 3, AXIS)
        text(ctx, sx(d), y_axis + 14, f"{d:+d}".replace("-", "\u2212") if d else "0", fs - 1, MUTED,
             align="center")
    text(ctx, x + w / 2, y_axis + 30, "Days relative to the mainshock", fs, INK_2, align="center")
    vline(ctx, sx(0), y, y_axis, INK, 1.0)
    text(ctx, sx(0) - 4, y + 2, f"Mw {case['mw']} mainshock" if case["mw"] else "mainshock",
         fs - 1, INK, align="right", valign="top")
    if t_cursor is not None:
        vline(ctx, sx(t_cursor), y, y_axis, hex_rgb("#e34948"), 1.6)
    return y_axis + 30


def render_heatmap(case, lim, path):
    W, H = 1200, 720
    surf, ctx = new_surface(W, H)
    text(ctx, 40, 36, f"{case['name']}: does the Lyapunov signal change beyond what the spectrum explains?",
         16, INK, bold=True)
    text(ctx, 40, 56, "Positive: windows more divergent than linear noise with the same spectrum; "
         "negative: less (e.g. a transient arrival).", 11, INK_2)
    bottom = draw_time_panels(ctx, case, (110, 100, 920, 520), lim)
    colorbar_h(ctx, 110, bottom + 30, 300, 10, lim, "Wolf minus surrogate mean (bits/s)")
    text(ctx, 1030, bottom + 50, f"{len(case['windows'])} windows, 39 IAAFT surrogates each",
         10, MUTED, align="right")
    surf.write_to_png(str(path))


# --- Map ----------------------------------------------------------------------

class MapProj:
    """Equirectangular map of the extent, widened in latitude or longitude to
    fill `box` = (x, y, w, h)."""

    def __init__(self, lat_min, lat_max, lon_min, lon_max, box):
        x, y, w, h = box
        k = math.cos(math.radians(0.5 * (lat_min + lat_max)))
        aspect = w / h
        span_x, span_y = (lon_max - lon_min) * k, lat_max - lat_min
        if span_x / span_y < aspect:
            grow = (span_y * aspect / k - (lon_max - lon_min)) / 2
            lon_min, lon_max = lon_min - grow, lon_max + grow
        else:
            grow = (span_x / aspect - span_y) / 2
            lat_min, lat_max = lat_min - grow, lat_max + grow
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = lat_min, lat_max, lon_min, lon_max
        sx = w / ((lon_max - lon_min) * k)
        sy = h / (lat_max - lat_min)
        self.s = min(sx, sy)  # px per degree of latitude
        self.k = k
        self.x0 = x + (w - (lon_max - lon_min) * k * self.s) / 2
        self.y0 = y + (h - (lat_max - lat_min) * self.s) / 2
        self.w = (lon_max - lon_min) * k * self.s
        self.h = (lat_max - lat_min) * self.s

    def __call__(self, lat, lon):
        return (self.x0 + (lon - self.lon_min) * self.k * self.s,
                self.y0 + (self.lat_max - lat) * self.s)


def map_extent(case, margin=0.25):
    lats = [float(s["lat"]) for s in case["stations"]] + [case["epi"][0]]
    lons = [float(s["lon"]) for s in case["stations"]] + [case["epi"][1]]
    return min(lats) - margin, max(lats) + margin, min(lons) - margin, max(lons) + margin


def draw_basemap(ctx, case, P, events=None, event_alpha=None, show_trace=True, fs=10):
    ctx.set_source_rgb(*PAGE)
    ctx.rectangle(P.x0, P.y0, P.w, P.h)
    ctx.fill()
    ctx.save()
    ctx.rectangle(P.x0, P.y0, P.w, P.h)
    ctx.clip()
    for g in range(math.ceil(P.lat_min * 2), math.floor(P.lat_max * 2) + 1):
        _, gy = P(g / 2, P.lon_min)
        hline(ctx, P.x0, P.x0 + P.w, gy, GRID)
    for g in range(math.ceil(P.lon_min * 2), math.floor(P.lon_max * 2) + 1):
        gx, _ = P(P.lat_min, g / 2)
        vline(ctx, gx, P.y0, P.y0 + P.h, GRID)
    for i, e in enumerate(events or []):
        a = 0.35 if event_alpha is None else event_alpha[i]
        if a <= 0.01:
            continue
        ex, ey = P(e["lat"], e["lon"])
        r = 0.6 + 0.55 * max(0.0, e["mag"] - 2.0)
        ctx.set_source_rgba(*INK_2, a)
        ctx.arc(ex, ey, r, 0, 2 * math.pi)
        ctx.fill()
    if show_trace and case["trace"]:
        (lo0, la0), (lo1, la1) = case["trace"]
        ctx.set_source_rgb(*INK)
        ctx.set_line_width(2.4)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(*P(la0, lo0))
        ctx.line_to(*P(la1, lo1))
        ctx.stroke()
    ctx.restore()
    ctx.set_source_rgb(*AXIS)
    ctx.set_line_width(1.0)
    ctx.rectangle(P.x0, P.y0, P.w, P.h)
    ctx.stroke()
    for g in range(math.ceil(P.lat_min * 2), math.floor(P.lat_max * 2) + 1):
        _, gy = P(g / 2, P.lon_min)
        text(ctx, P.x0 - 4, gy, f"{g / 2:.1f}°N", fs - 1.5, MUTED, align="right", valign="middle")
    for g in range(math.ceil(P.lon_min * 2), math.floor(P.lon_max * 2) + 1):
        gx, _ = P(P.lat_min, g / 2)
        text(ctx, gx, P.y0 + P.h + 13, f"{g / 2:.1f}°E", fs - 1.5, MUTED, align="center")
    # scale bar: 25 km
    km = 25.0
    px = km / 111.195 * P.s
    bx, by = P.x0 + 12, P.y0 + P.h - 14
    hline(ctx, bx, bx + px, by, INK, 2.0)
    text(ctx, bx + px / 2, by - 5, f"{km:.0f} km", fs - 1.5, INK_2, align="center")


def star(ctx, cx, cy, r, color=INK, ring=SURFACE):
    ctx.new_path()
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        a = -math.pi / 2 + i * math.pi / 5
        ctx.line_to(cx + rr * math.cos(a), cy + rr * math.sin(a))
    ctx.close_path()
    ctx.set_source_rgb(*color)
    ctx.fill_preserve()
    ctx.set_source_rgb(*ring)
    ctx.set_line_width(1.5)
    ctx.stroke()


def station_marker(ctx, x, y, fill, significant, r=10):
    ctx.arc(x, y, r + 2, 0, 2 * math.pi)  # surface ring separates it from the events
    ctx.set_source_rgb(*SURFACE)
    ctx.fill()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.set_source_rgb(*fill)
    ctx.fill_preserve()
    ctx.set_source_rgb(*(INK if significant else MUTED))
    ctx.set_line_width(2.2 if significant else 1.0)
    ctx.stroke()


def render_station_map(case, path):
    W, H = 1200, 690
    surf, ctx = new_surface(W, H)
    text(ctx, 40, 36, f"{case['name']}: post − pre change of the broadband Wolf exponent, by station",
         16, INK, bold=True)
    text(ctx, 40, 56, "Same color scale in both panels. Bold ring: 95% day-block bootstrap interval excludes 0.",
         11, INK_2)
    post_days = round(max(w["rel"] for w in case["windows"]))
    evs = [e for e in case["events"] if 0 <= e["rel"] <= post_days and e["mag"] >= 2.5]
    vals = [abs(float(s["delta_wolf"])) for s in case["stations"] if s["delta_wolf"]]
    lim = math.ceil(max(vals + [0.05]) * 10) / 10
    lat0, lat1, lon0, lon1 = map_extent(case)
    panels = [
        ("A. Raw change in the Wolf exponent", "delta_wolf"),
        ("B. Change not explained by the spectrum (excess over surrogates)", "delta_excess"),
    ]
    for p, (title, key) in enumerate(panels):
        box = (60 + p * 575, 100, 520, 470)
        P = MapProj(lat0, lat1, lon0, lon1, box)
        text(ctx, P.x0, 88, title, 12, INK, bold=True)
        draw_basemap(ctx, case, P, evs)
        ex, ey = P(*case["epi"])
        star(ctx, ex, ey, 9)
        for s in case["stations"]:
            if not s[key]:
                continue
            v = float(s[key])
            lo, hi = float(s[key + "_ci_lo"] or "nan"), float(s[key + "_ci_hi"] or "nan")
            sig = (lo > 0 or hi < 0)
            x, y = P(float(s["lat"]), float(s["lon"]))
            station_marker(ctx, x, y, diverging(v, lim), sig)
            left = x > P.x0 + 0.62 * P.w
            tx, al = (x - 15, "right") if left else (x + 15, "left")
            text(ctx, tx, y - 2, f"{s['station']}  {float(s['dist_km']):.0f} km", 10.5, INK,
                 bold=True, align=al)
            text(ctx, tx, y + 11, f"{signed(v)}  [{signed(lo)}, {signed(hi)}]", 10, INK_2, align=al)
    # legend row
    ly = 618
    colorbar_h(ctx, 60, ly, 300, 10, lim, "Post − pre mean Wolf exponent (bits/s)")
    lx = 430
    star(ctx, lx, ly + 4, 7)
    text(ctx, lx + 12, ly + 8, "Epicentre", 10.5, INK_2)
    hline(ctx, lx + 90, lx + 116, ly + 4, INK, 2.4)
    text(ctx, lx + 122, ly + 8, "Inverted fault (upper edge)", 10.5, INK_2)
    ctx.set_source_rgba(*INK_2, 0.35)
    ctx.arc(lx + 300, ly + 4, 2.0, 0, 2 * math.pi)
    ctx.fill()
    text(ctx, lx + 308, ly + 8, f"Aftershocks M 2.5+, first {post_days} d", 10.5, INK_2)
    station_marker(ctx, lx + 10, ly + 34, NEUTRAL, True, 7)
    text(ctx, lx + 24, ly + 38, "Interval excludes 0", 10.5, INK_2)
    station_marker(ctx, lx + 160, ly + 34, NEUTRAL, False, 7)
    text(ctx, lx + 174, ly + 38, "Interval includes 0", 10.5, INK_2)
    rel = [w["rel"] for w in case["windows"]]
    text(ctx, W - 40, H - 14, f"Pre: {-min(rel):.0f} d before, post: {max(rel):.0f} d after the "
         "mainshock; values are station means of 10-min windows.", 9.5, MUTED, align="right")
    surf.write_to_png(str(path))


# --- Video ----------------------------------------------------------------------

def render_video(case, lim, path, fps=30):
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")
    W, H = 960, 540
    lat0, lat1, lon0, lon1 = map_extent(case, 0.3)
    P = MapProj(lat0, lat1, lon0, lon1, (46, 52, 440, 396))
    by_k = defaultdict(dict)
    for w in case["windows"]:
        by_k[w["k"]][w["station"]] = w
    events = sorted(case["events"], key=lambda e: e["rel"])
    fade_days = 1.0
    k0 = math.ceil(case["t0"] / STRIDE_DAYS)
    k1 = math.floor(case["t1"] / STRIDE_DAYS)

    with tempfile.TemporaryDirectory() as tmp:
        for f, k in enumerate(range(k0, k1 + 1)):
            t = k * STRIDE_DAYS
            surf, ctx = new_surface(W, H)
            text(ctx, 46, 30, f"{case['name']}: broadband Lyapunov signal vs. its surrogates",
                 13, INK, bold=True)
            day_txt = signed(t, 1) if abs(t) >= 0.05 else "0.0"
            text(ctx, 936, 30, f"Day {day_txt}", 13, INK, bold=True, align="right")
            text(ctx, 936, 46, fmt_date(case, t), 9.5, INK_2, align="right")
            alpha = []
            for e in events:
                age = t - e["rel"]
                alpha.append(0.0 if age < 0 or age > fade_days or e["mag"] < 2.0
                             else 0.55 * (1 - age / fade_days))
            draw_basemap(ctx, case, P, events, alpha, show_trace=t >= 0, fs=9)
            if t >= 0:
                ex, ey = P(*case["epi"])
                star(ctx, ex, ey, 8)
            for s in case["stations"]:
                x, y = P(float(s["lat"]), float(s["lon"]))
                w = by_k[k].get(s["station"])
                if w is None:
                    station_marker(ctx, x, y, PAGE, False, 8)
                    text(ctx, x + 12, y + 4, f"{s['station']}  no window", 9, MUTED)
                    continue
                station_marker(ctx, x, y, diverging(w["excess"], lim), w["outside"], 8)
                if w["outside"]:
                    ctx.arc(x, y, 14, 0, 2 * math.pi)
                    ctx.set_source_rgb(*INK)
                    ctx.set_line_width(1.2)
                    ctx.stroke()
                text(ctx, x + 16, y + 4, f"{s['station']}  {signed(w['excess'], 3)}", 9, INK)
            draw_time_panels(ctx, case, (566, 84, 370, 352), lim, t_cursor=t, compact=True)
            colorbar_h(ctx, 46, 494, 220, 7, lim, "Wolf minus surrogate mean (bits/s)")
            text(ctx, 300, 500, "Ring: window outside all 39 surrogates", 9, INK_2)
            text(ctx, 300, 515, "Gray dots: catalogued earthquakes, fading over 24 h", 9, INK_2)
            text(ctx, 566, 515, "Red line: current time", 9, INK_2)
            surf.write_to_png(f"{tmp}/f{f:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", f"{tmp}/f%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", str(path)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--tag", default="", help="lseis-study output tag (e.g. long_mc)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    case = load_case(args.config, args.tag)
    lim = excess_limit(case["windows"])
    out, sfx = case["out_dir"], case["sfx"]
    render_heatmap(case, lim, out / f"lseis_heatmap{sfx}.png")
    render_station_map(case, out / f"lseis_station_map{sfx}.png")
    print(f"[plot_lseis] Wrote {out}/lseis_heatmap{sfx}.png, lseis_station_map{sfx}.png")
    if args.video:
        render_video(case, lim, out / f"lseis_timelapse{sfx}.mp4", args.fps)
        print(f"[plot_lseis] Wrote {out}/lseis_timelapse{sfx}.mp4")


if __name__ == "__main__":
    main()
