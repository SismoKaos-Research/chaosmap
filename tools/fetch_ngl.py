#!/usr/bin/env python3
"""Download NGL (Nevada Geodetic Laboratory) daily GNSS time series for a region.

Fetches the NGL station index (DataHoldings.txt) and the equipment/earthquake
step log (steps.txt), selects stations inside a lat/lon box, and downloads one
.tenv3 file per station. Standard library only.

Re-running is safe: a station is skipped when its file already exists and is
not older than the station's last solution date (Dtmod) in DataHoldings.

    python3 tools/fetch_ngl.py                      # Turkey box, into data/gnss
    python3 tools/fetch_ngl.py --dry-run            # list what would be fetched
    python3 tools/fetch_ngl.py --box 36 42 26 45    # lat_min lat_max lon_min lon_max
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

BASE = "https://geodesy.unr.edu"
HOLDINGS_URL = f"{BASE}/NGLStationPages/DataHoldings.txt"
STEPS_URL = f"{BASE}/NGLStationPages/steps.txt"
# Tried in order. IGS20 is the current NGL reference frame; IGS14 is the legacy
# product (what gnssdl's files appear to be) and is kept as a fallback.
TENV3_URLS = {
    "IGS20": f"{BASE}/gps_timeseries/IGS20/tenv3/IGS20/{{sta}}.tenv3",
    "IGS14": f"{BASE}/gps_timeseries/tenv3/IGS14/{{sta}}.tenv3",
}
# Turkey plus margins (Aegean islands, Cyprus, southern Caucasus), so that the
# strain/velocity interpolation is constrained at the edges of the country.
DEFAULT_BOX = (35.0, 43.0, 25.0, 45.0)
USER_AGENT = "chaosmap-fetch/0.1 (research; NGL tenv3)"


@dataclass(frozen=True)
class Station:
    sta: str
    lat: float
    lon: float
    height: float
    dtbeg: str
    dtend: str
    dtmod: str
    nsol: int


def http_get(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2.0 * (attempt + 1))
    assert last is not None
    raise last


def wrap_lon(lon: float) -> float:
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


def parse_holdings(text: str) -> list[Station]:
    out: list[Station] = []
    for line in text.splitlines()[1:]:
        f = line.split()
        if len(f) < 11:
            continue
        try:
            out.append(Station(
                sta=f[0], lat=float(f[1]), lon=wrap_lon(float(f[2])),
                height=float(f[3]), dtbeg=f[7], dtend=f[8], dtmod=f[9],
                nsol=int(f[10]),
            ))
        except ValueError:
            continue
    return out


def needs_fetch(path: Path, st: Station) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    try:
        mod = dt.datetime.strptime(st.dtmod, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return False
    have = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return have < mod


def fetch_station(st: Station, outdir: Path, frames: list[str]) -> tuple[str, str, int]:
    """Returns (station, frame or status, bytes)."""
    for frame in frames:
        path = outdir / frame / f"{st.sta}.tenv3"
        if not needs_fetch(path, st):
            return st.sta, f"{frame}:cached", path.stat().st_size
        try:
            data = http_get(TENV3_URLS[frame].format(sta=st.sta))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            return st.sta, f"error:{e}", 0
        except Exception as e:  # noqa: BLE001 - report and move on
            return st.sta, f"error:{e}", 0
        if not data.lstrip().startswith(b"site"):
            return st.sta, "error:unexpected-content", 0
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return st.sta, frame, len(data)
    return st.sta, "missing", 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/gnss"))
    ap.add_argument("--box", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                    default=DEFAULT_BOX)
    ap.add_argument("--min-solutions", type=int, default=365,
                    help="skip stations with fewer daily solutions than this")
    ap.add_argument("--frames", nargs="+", default=["IGS20", "IGS14"], choices=list(TENV3_URLS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="fetch at most N stations (testing)")
    args = ap.parse_args()

    meta = args.out / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    print(f"fetching {HOLDINGS_URL}", file=sys.stderr)
    holdings = http_get(HOLDINGS_URL).decode("utf-8", "replace")
    (meta / "DataHoldings.txt").write_text(holdings)
    print(f"fetching {STEPS_URL}", file=sys.stderr)
    (meta / "steps.txt").write_bytes(http_get(STEPS_URL))

    la0, la1, lo0, lo1 = args.box
    stations = [s for s in parse_holdings(holdings)
                if la0 <= s.lat <= la1 and lo0 <= s.lon <= lo1 and s.nsol >= args.min_solutions]
    stations.sort(key=lambda s: s.sta)
    if args.limit:
        stations = stations[: args.limit]
    print(f"{len(stations)} stations in box {args.box} with >= {args.min_solutions} solutions",
          file=sys.stderr)

    if args.dry_run:
        for s in stations:
            print(f"{s.sta} {s.lat:9.4f} {s.lon:9.4f} {s.dtbeg} {s.dtend} {s.nsol}")
        return 0

    results: dict[str, tuple[str, int]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_station, s, args.out, args.frames): s for s in stations}
        for i, fut in enumerate(as_completed(futs), 1):
            sta, status, nbytes = fut.result()
            results[sta] = (status, nbytes)
            print(f"[{i:4d}/{len(stations)}] {sta} {status} {nbytes / 1e6:.2f} MB", file=sys.stderr)

    with open(args.out / "stations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "latitude", "longitude", "height_m", "dtbeg", "dtend", "nsol", "frame", "bytes"])
        for s in stations:
            status, nbytes = results.get(s.sta, ("missing", 0))
            w.writerow([s.sta, s.lat, s.lon, s.height, s.dtbeg, s.dtend, s.nsol, status.split(":")[0], nbytes])

    bad = {k: v for k, v in results.items() if v[0].startswith(("error", "missing"))}
    total = sum(n for _, n in results.values())
    print(f"done: {len(results) - len(bad)} ok, {len(bad)} failed/missing, {total / 1e6:.1f} MB",
          file=sys.stderr)
    for k, (status, _) in sorted(bad.items()):
        print(f"  {k}: {status}", file=sys.stderr)
    return 1 if len(bad) == len(results) and results else 0


if __name__ == "__main__":
    sys.exit(main())
