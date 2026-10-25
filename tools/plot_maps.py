#!/usr/bin/env python3
"""Renders publication-grade 4-panel Before / Moved / After / Change fault maps
from chaosmap output directories using PyCairo (PNG + SVG, zero numpy/matplotlib
dependency).
"""

import csv
import json
import math
import pathlib
import sys
import cairo


def viridis_rgb(t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))
    stops = [
        (0.267, 0.005, 0.329),
        (0.283, 0.141, 0.458),
        (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553),
        (0.164, 0.471, 0.558),
        (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518),
        (0.267, 0.749, 0.441),
        (0.478, 0.821, 0.318),
        (0.741, 0.873, 0.150),
        (0.993, 0.906, 0.144),
    ]
    x = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(x))
    f = x - i
    a, b = stops[i], stops[i + 1]
    return (
        a[0] + f * (b[0] - a[0]),
        a[1] + f * (b[1] - a[1]),
        a[2] + f * (b[2] - a[2]),
    )


def inferno_rgb(t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))
    stops = [
        (0.05, 0.04, 0.15),
        (0.26, 0.04, 0.41),
        (0.58, 0.15, 0.40),
        (0.87, 0.32, 0.23),
        (0.99, 0.65, 0.04),
        (0.99, 0.98, 0.64),
    ]
    x = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(x))
    f = x - i
    a, b = stops[i], stops[i + 1]
    return (
        a[0] + f * (b[0] - a[0]),
        a[1] + f * (b[1] - a[1]),
        a[2] + f * (b[2] - a[2]),
    )


def diverging_bwr_rgb(v: float) -> tuple[float, float, float]:
    # v in [-1, 1]
    v = max(-1.0, min(1.0, v))
    if v < 0.0:
        t = 1.0 + v  # 0 at -1, 1 at 0
        return (0.10 + 0.85 * t, 0.25 + 0.70 * t, 0.75 + 0.20 * t)
    t = v  # 0 at 0, 1 at +1
    return (0.95, 0.95 - 0.75 * t, 0.95 - 0.80 * t)


def draw_arrow(
    ctx: cairo.Context,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    head_len: float = 7.0,
) -> None:
    ctx.move_to(x0, y0)
    ctx.line_to(x1, y1)
    ctx.stroke()
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L < 2.0:
        return
    ang = math.atan2(dy, dx)
    a1 = ang + math.pi - 0.45
    a2 = ang + math.pi + 0.45
    ctx.move_to(x1, y1)
    ctx.line_to(x1 + head_len * math.cos(a1), y1 + head_len * math.sin(a1))
    ctx.line_to(x1 + head_len * math.cos(a2), y1 + head_len * math.sin(a2))
    ctx.close_path()
    ctx.fill()


def render_case(out_dir: pathlib.Path) -> None:
    metrics = json.loads((out_dir / "summary_metrics.json").read_text())
    geojson = json.loads((out_dir / "fault_features.geojson").read_text())

    with (out_dir / "grids.csv").open() as f:
        grid_rows = list(csv.DictReader(f))
    with (out_dir / "stations_gnss.csv").open() as f:
        gnss_rows = list(csv.DictReader(f))

    seis_rows = []
    seis_csv = out_dir / "lyapunov_stations.csv"
    if seis_csv.exists():
        with seis_csv.open() as f:
            seis_rows = list(csv.DictReader(f))

    lats = sorted({float(r["lat"]) for r in grid_rows})
    lons = sorted({float(r["lon"]) for r in grid_rows})
    lat_min, lat_max = lats[0], lats[-1]
    lon_min, lon_max = lons[0], lons[-1]
    dlat = lats[1] - lats[0] if len(lats) > 1 else 0.05
    dlon = lons[1] - lons[0] if len(lons) > 1 else 0.05

    W, H = 1800, 1320
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)

    # Dark slate background
    ctx.set_source_rgb(0.07, 0.09, 0.13)
    ctx.paint()

    # Header banner
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(26)
    ctx.set_source_rgb(0.96, 0.97, 0.99)
    title = (
        f"CHAOSMAP FAULT KINEMATICS & LYAPUNOV FUSION — "
        f"{metrics['event_name']} ({metrics['date_iso']})"
    )
    ctx.move_to(40, 42)
    ctx.show_text(title)

    hm = metrics["how_much_it_moved"]
    fg = metrics["coseismic_fault_geometry"]
    cmp_m = metrics["before_vs_after_fault_comparison"]
    bm = metrics["lyapunov_benchmark"]

    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(15)
    ctx.set_source_rgb(0.78, 0.84, 0.92)
    subtitle = (
        f"Inverted Fault: Strike {fg['strike_deg']:.1f}°, Dip {fg['dip_deg']:.1f}°, "
        f"Rake {fg['mean_rake_deg']:.1f}° ({fg['length_km']:.0f}×{fg['width_km']:.0f} km)  |  "
        f"Peak Slip: {hm['peak_fault_slip_m']:.2f} m (Geodetic Mw {hm['geodetic_mw']:.2f}, VR {hm['variance_reduction_pct']:.1f}%)  |  "
        f"Max GNSS Step: {hm['max_disp_station']} ({hm['max_station_horiz_mm']:.1f} mm)  |  "
        f"Mode: {bm['selected_mode']}"
    )
    ctx.move_to(40, 68)
    ctx.show_text(subtitle)

    panels = [
        (
            60,
            115,
            790,
            515,
            "A. BEFORE EARTHQUAKE: Fault Susceptibility (L-geo FTLE + L-seis + Strain + Pre-Seismicity)",
            "before_susceptibility",
            "viridis",
        ),
        (
            940,
            115,
            790,
            515,
            "B. HOW MUCH IT MOVED: Coseismic Distributed Slip (m), Okada Field & GNSS Offsets",
            "moved_intensity",
            "inferno",
        ),
        (
            60,
            715,
            790,
            515,
            "C. AFTER EARTHQUAKE: Activated Rupture Zone, 3D Aftershock Plane & Post-Lyapunov",
            "after_state",
            "viridis",
        ),
        (
            940,
            715,
            790,
            515,
            "D. BEFORE vs AFTER CHANGE: Δ Fault State (After − Before) & Geometry Shift",
            "change_after_minus_before",
            "bwr",
        ),
    ]

    for px, py, pw, ph, ptitle, col_name, cmap in panels:
        def proj(lat: float, lon: float) -> tuple[float, float]:
            u = (lon - lon_min) / max(1e-9, lon_max - lon_min)
            v = (lat - lat_min) / max(1e-9, lat_max - lat_min)
            return px + u * pw, py + (1.0 - v) * ph

        # Panel title
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(15)
        ctx.set_source_rgb(0.94, 0.96, 1.0)
        ctx.move_to(px, py - 12)
        ctx.show_text(ptitle)

        ctx.save()
        ctx.rectangle(px, py, pw, ph)
        ctx.clip()

        # 1. Raster heatmap cells
        cw = pw * (dlon / max(1e-9, lon_max - lon_min)) + 1.0
        ch = ph * (dlat / max(1e-9, lat_max - lat_min)) + 1.0
        for r in grid_rows:
            lat = float(r["lat"])
            lon = float(r["lon"])
            val = float(r[col_name])
            x, y = proj(lat + 0.5 * dlat, lon - 0.5 * dlon)
            if cmap == "viridis":
                cr, cg, cb = viridis_rgb(val)
            elif cmap == "inferno":
                cr, cg, cb = inferno_rgb(val)
            else:
                cr, cg, cb = diverging_bwr_rgb(val)
            ctx.set_source_rgb(cr, cg, cb)
            ctx.rectangle(x, y, cw, ch)
            ctx.fill()

        # Subtle lat/lon graticule
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.14)
        ctx.set_line_width(0.8)
        for glat in range(int(math.ceil(lat_min)), int(math.floor(lat_max)) + 1):
            x0, y0 = proj(glat, lon_min)
            x1, y1 = proj(glat, lon_max)
            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.stroke()
        for glon in range(int(math.ceil(lon_min)), int(math.floor(lon_max)) + 1):
            x0, y0 = proj(lat_min, glon)
            x1, y1 = proj(lat_max, glon)
            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.stroke()

        # 2. Panel-specific overlays from GeoJSON & stations
        max_slip = max(0.1, hm["peak_fault_slip_m"])
        if col_name == "moved_intensity":
            # Draw distributed slip sub-patches colored by slip_m
            for feat in geojson["features"]:
                props = feat["properties"]
                if props.get("role") == "slip_patch":
                    ring = feat["geometry"]["coordinates"][0]
                    s_frac = float(props["slip_m"]) / max_slip
                    cr, cg, cb = inferno_rgb(min(1.0, 0.2 + 0.8 * s_frac))
                    ctx.move_to(*proj(ring[0][1], ring[0][0]))
                    for pt in ring[1:]:
                        ctx.line_to(*proj(pt[1], pt[0]))
                    ctx.close_path()
                    ctx.set_source_rgba(cr, cg, cb, 0.72)
                    ctx.fill_preserve()
                    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.45)
                    ctx.set_line_width(1.0)
                    ctx.stroke()

        # Draw fault traces / ridges
        for feat in geojson["features"]:
            props = feat["properties"]
            role = props.get("role", "")
            if feat["geometry"]["type"] != "LineString":
                continue
            c0, c1 = feat["geometry"]["coordinates"]
            x0, y0 = proj(c0[1], c0[0])
            x1, y1 = proj(c1[1], c1[0])

            if role == "coseismic_fault_trace" and col_name in (
                "moved_intensity",
                "after_state",
                "change_after_minus_before",
            ):
                ctx.set_dash([])
                ctx.set_source_rgb(1.0, 0.18, 0.28)
                ctx.set_line_width(3.8)
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.stroke()
            elif role == "pre_seismicity_trace" and col_name in (
                "before_susceptibility",
                "change_after_minus_before",
            ):
                ctx.set_dash([8.0, 5.0])
                ctx.set_source_rgb(0.20, 0.92, 1.0)
                ctx.set_line_width(2.6)
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.stroke()
                ctx.set_dash([])
            elif role == "l_geo_ftle_ridge" and col_name == "before_susceptibility":
                ctx.set_dash([4.0, 4.0])
                ctx.set_source_rgb(1.0, 0.85, 0.20)
                ctx.set_line_width(2.4)
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.stroke()
                ctx.set_dash([])
            elif role == "post_seismicity_trace" and col_name in (
                "after_state",
                "change_after_minus_before",
            ):
                ctx.set_dash([])
                ctx.set_source_rgb(1.0, 0.82, 0.15)
                ctx.set_line_width(2.8)
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.stroke()
            elif role == "aftershock_segment" and col_name == "after_state":
                ctx.set_dash([5.0, 3.0])
                ctx.set_source_rgba(1.0, 0.95, 0.65, 0.85)
                ctx.set_line_width(2.0)
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.stroke()
                ctx.set_dash([])

        # Hypocentre star/circle
        hx, hy = proj(metrics["hypocentre"]["lat"], metrics["hypocentre"]["lon"])
        ctx.arc(hx, hy, 7.0, 0, 2 * math.pi)
        ctx.set_source_rgb(1.0, 0.15, 0.15)
        ctx.fill_preserve()
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.set_line_width(1.8)
        ctx.stroke()

        # GNSS vectors
        if col_name == "before_susceptibility":
            # Interseismic velocity arrows (mm/yr)
            ctx.set_font_size(10)
            for s in gnss_rows:
                if int(s["has_vel"]) == 0:
                    continue
                sx, sy = proj(float(s["lat"]), float(s["lon"]))
                ve = float(s["ve_mm_yr"])
                vn = float(s["vn_mm_yr"])
                scale = 2.2  # pixels per mm/yr
                ctx.set_source_rgb(0.95, 0.98, 1.0)
                ctx.set_line_width(1.6)
                draw_arrow(ctx, sx, sy, sx + ve * scale, sy - vn * scale, 5.5)
        elif col_name == "moved_intensity":
            # Coseismic step arrows
            max_co_mm = max(10.0, hm["max_station_horiz_mm"])
            scale = 85.0 / max_co_mm
            ctx.set_font_size(11)
            for s in gnss_rows:
                if int(s["has_coseismic"]) == 0:
                    continue
                sx, sy = proj(float(s["lat"]), float(s["lon"]))
                ce = float(s["co_e_mm"])
                cn = float(s["co_n_mm"])
                ch_mm = float(s["co_horiz_mm"])
                ctx.set_source_rgb(0.15, 0.98, 0.95)
                ctx.set_line_width(2.2)
                draw_arrow(ctx, sx, sy, sx + ce * scale, sy - cn * scale, 6.5)
                if ch_mm >= 0.12 * max_co_mm:
                    ctx.set_source_rgb(1.0, 1.0, 1.0)
                    ctx.move_to(sx + 5, sy - 5)
                    ctx.show_text(f"{s['station']} ({ch_mm:.0f}mm)")

        # Broadband seismic stations on Panel A & C
        if col_name in ("before_susceptibility", "after_state") and seis_rows:
            ctx.set_font_size(11)
            for sr in seis_rows:
                sx, sy = proj(float(sr["lat"]), float(sr["lon"]))
                ctx.move_to(sx, sy - 6)
                ctx.line_to(sx - 6, sy + 5)
                ctx.line_to(sx + 6, sy + 5)
                ctx.close_path()
                ctx.set_source_rgb(0.15, 1.0, 0.45)
                ctx.fill_preserve()
                ctx.set_source_rgb(0.0, 0.0, 0.0)
                ctx.set_line_width(1.2)
                ctx.stroke()
                ctx.set_source_rgb(0.95, 1.0, 0.95)
                ctx.move_to(sx + 8, sy + 4)
                dw = float(sr["delta_wolf"])
                ctx.show_text(f"{sr['station']} (Δλ={dw:+.2f})")

        # Summary callout box on Panel D
        if col_name == "change_after_minus_before":
            bx, by, bw, bh = px + 16, py + 16, 355, 138
            ctx.set_source_rgba(0.05, 0.07, 0.11, 0.84)
            ctx.rectangle(bx, by, bw, bh)
            ctx.fill_preserve()
            ctx.set_source_rgba(0.8, 0.85, 0.95, 0.5)
            ctx.set_line_width(1.2)
            ctx.stroke()

            ctx.set_font_size(12)
            ctx.set_source_rgb(0.96, 0.98, 1.0)
            lines = [
                f"Before Strike: {cmp_m['before_strike_deg']:.1f}° → Coseismic: {cmp_m['coseismic_strike_deg']:.1f}°",
                f"Aftershock Plane: {cmp_m['after_strike_deg']:.1f}° (Dip {cmp_m['post_aftershock_dip_deg']:.1f}°, L={cmp_m['post_aftershock_length_km']:.0f} km)",
                f"Strike Rotation Δ: {cmp_m['strike_change_deg']:.1f}° | Centroid Shift: {cmp_m['centroid_shift_km']:.1f} km",
                f"Peak Slip: {hm['peak_fault_slip_m']:.2f} m | Top-25% Slip: {hm['top_quartile_fault_slip_m']:.2f} m",
                f"L-geo Skill: {bm['l_geo']['composite_skill']:.2f} (w={bm['l_geo']['assigned_weight']:.2f})",
                f"L-seis Skill: {bm['l_seis']['composite_skill']:.2f} (w={bm['l_seis']['assigned_weight']:.2f}, AUC={bm['l_seis']['temporal_pre_auc']:.2f})",
            ]
            for idx_l, ln in enumerate(lines):
                ctx.move_to(bx + 12, by + 22 + idx_l * 20)
                ctx.show_text(ln)

        ctx.restore()

        # Panel border and axis labels
        ctx.set_source_rgb(0.75, 0.80, 0.88)
        ctx.set_line_width(1.5)
        ctx.rectangle(px, py, pw, ph)
        ctx.stroke()

        ctx.set_font_size(11)
        ctx.set_source_rgb(0.75, 0.80, 0.88)
        for glat in range(int(math.ceil(lat_min)), int(math.floor(lat_max)) + 1):
            _, y0 = proj(glat, lon_min)
            ctx.move_to(px - 38, y0 + 4)
            ctx.show_text(f"{glat}°N")
        for glon in range(int(math.ceil(lon_min)), int(math.floor(lon_max)) + 1):
            x0, _ = proj(lat_min, glon)
            ctx.move_to(x0 - 14, py + ph + 18)
            ctx.show_text(f"{glon}°E")

    png_path = out_dir / "fault_map_4panel.png"
    surf.write_to_png(str(png_path))
    print(f"[plot_maps] Wrote {png_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_maps.py <out_case_dir> [...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        render_case(pathlib.Path(arg))
