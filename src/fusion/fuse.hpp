#pragma once

// Fuses Geodesy (GNSS velocities, strain rate, Okada coseismic slip),
// Seismicity (pre-event background planes, post-event aftershock planes), and
// Lyapunov fields (`L-geo` FTLE and `L-seis` broadband Wolf/Rosenstein) into:
// 1. BEFORE fault susceptibility map & lineament
// 2. MOVED coseismic slip & surface displacement map
// 3. AFTER activated fault zone map & aftershock planes
// 4. CHANGE (After vs Before) map & quantitative comparison metrics.

#include "anomaly/array_anomaly.hpp"
#include "fusion/benchmark.hpp"
#include "geo/projection.hpp"
#include "gnss/fields.hpp"
#include "gnss/trajectory.hpp"
#include "okada/inversion.hpp"
#include "okada/okada85.hpp"
#include "seis/planes.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace chaosmap::fusion {

struct FusedFaultMap {
  // 2D grids on `grid`
  std::vector<double> before_susceptibility;     // [0, 1]
  std::vector<double> moved_surface_disp_m;      // Forward Okada |u_horiz| (m)
  std::vector<double> moved_intensity;           // [0, 1]
  std::vector<double> after_state;               // [0, 1]
  std::vector<double> change_after_minus_before; // [-1, 1]

  // Summary "Before -> Moved -> After -> Change" fault metrics
  std::string max_disp_station;
  double max_station_horiz_m{};
  double max_station_up_m{};

  double peak_slip_m{};
  double mean_slip_m{};
  double top_quartile_slip_m{};
  double geodetic_mw{};
  double variance_reduction_pct{};

  double before_strike_deg{};
  double coseismic_strike_deg{};
  double after_strike_deg{};
  double strike_change_deg{};
  double centroid_shift_km{};
};

namespace detail {

[[nodiscard]] inline std::vector<double>
normalize_01(std::span<const double> v) {
  std::vector<double> out(v.begin(), v.end());
  double mn = 1e30, mx = -1e30;
  for (double x : out) {
    if (std::isfinite(x)) {
      mn = std::min(mn, x);
      mx = std::max(mx, x);
    }
  }
  const double span = mx - mn;
  if (span > 1e-12) {
    for (double &x : out)
      x = std::clamp((x - mn) / span, 0.0, 1.0);
  } else {
    std::ranges::fill(out, 0.0);
  }
  return out;
}

} // namespace detail

[[nodiscard]] inline FusedFaultMap fuse_fault_states(
    const geo::Grid2D &grid, std::span<const gnss::StationMotion> gnss_stations,
    const gnss::GeodeticGridFields &geod, const okada::InversionResult &inv,
    const seis::SeismicityAnalysis &pre_seis,
    const seis::SeismicityAnalysis &post_seis,
    const anomaly::ArrayAnomalyResult &seis_anom, const LyapunovBenchmark &bm) {
  FusedFaultMap out;
  const std::size_t N = grid.size();
  out.before_susceptibility.assign(N, 0.0);
  out.moved_surface_disp_m.assign(N, 0.0);
  out.moved_intensity.assign(N, 0.0);
  out.after_state.assign(N, 0.0);
  out.change_after_minus_before.assign(N, 0.0);

  // 1. Station coseismic maximum
  for (const auto &s : gnss_stations) {
    if (!s.has_coseismic)
      continue;
    const double h = std::hypot(s.coseismic_enu_m.x, s.coseismic_enu_m.y);
    if (h > out.max_station_horiz_m) {
      out.max_station_horiz_m = h;
      out.max_station_up_m = s.coseismic_enu_m.z;
      out.max_disp_station = s.station;
    }
  }

  // 2. Distributed slip statistics
  out.peak_slip_m = inv.max_slip_m;
  out.geodetic_mw = inv.mw_geodetic;
  out.variance_reduction_pct = inv.variance_reduction_pct;
  if (!inv.patches.empty()) {
    std::vector<double> slips;
    slips.reserve(inv.patches.size());
    double sum_s = 0.0;
    for (const auto &p : inv.patches) {
      slips.push_back(p.total_slip_m);
      sum_s += p.total_slip_m;
    }
    out.mean_slip_m = sum_s / static_cast<double>(slips.size());
    std::ranges::sort(slips, std::greater<>{});
    const std::size_t q_n = std::max<std::size_t>(1, slips.size() / 4);
    double q_sum = 0.0;
    for (std::size_t i = 0; i < q_n; ++i)
      q_sum += slips[i];
    out.top_quartile_slip_m = q_sum / static_cast<double>(q_n);
  }

  // 3. Forward Okada surface displacement field on `grid` from distributed
  // slip patches. Patch centroids are stored in the inversion's
  // (epicentre-centred) projection, so re-express them in the grid projection
  // through their geographic coordinates first.
  std::vector<okada::FaultPatch> grid_patches;
  grid_patches.reserve(inv.patches.size());
  for (const auto &sp : inv.patches) {
    okada::FaultPatch p = sp.patch;
    const auto enu = grid.proj.to_enu(sp.lat_deg, sp.lon_deg);
    p.east_km = enu.x;
    p.north_km = enu.y;
    grid_patches.push_back(p);
  }
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const std::size_t idx = r * grid.n_lon + c;
      const auto pt = grid.enu_at(r, c);
      linalg::Vec3 u_tot{};
      for (const auto &p : grid_patches) {
        u_tot = u_tot + okada::dislocation_enu(p, pt.x, pt.y);
      }
      out.moved_surface_disp_m[idx] = std::hypot(u_tot.x, u_tot.y);
    }
  }

  const auto norm_shear = detail::normalize_01(geod.max_shear_nanostrain_yr);
  const auto norm_moved_u = detail::normalize_01(out.moved_surface_disp_m);

  const double w_geo = bm.l_geo.assigned_weight;
  const double w_seis = bm.l_seis.assigned_weight;

  // 4. Fuse BEFORE, MOVED, AFTER, and CHANGE grids
  for (std::size_t i = 0; i < N; ++i) {
    const double l_before =
        w_geo * geod.ftle_interseismic[i] +
        w_seis * (seis_anom.has_seismic ? seis_anom.grid_l_seis_pre[i] : 0.0);

    out.before_susceptibility[i] = 0.45 * l_before + 0.30 * norm_shear[i] +
                                   0.25 * pre_seis.grid_density[i];

    out.moved_intensity[i] =
        0.70 * norm_moved_u[i] + 0.30 * geod.ftle_coseismic[i];

    const double l_after =
        w_geo * geod.ftle_coseismic[i] +
        w_seis * (seis_anom.has_seismic ? seis_anom.grid_l_seis_post[i] : 0.0);

    out.after_state[i] = 0.45 * post_seis.grid_density[i] + 0.30 * l_after +
                         0.25 * norm_moved_u[i];
  }

  out.before_susceptibility = detail::normalize_01(out.before_susceptibility);
  out.moved_intensity = detail::normalize_01(out.moved_intensity);
  out.after_state = detail::normalize_01(out.after_state);

  for (std::size_t i = 0; i < N; ++i) {
    out.change_after_minus_before[i] =
        out.after_state[i] - out.before_susceptibility[i];
  }

  // 5. Strike & centroid shift comparison (Before vs Coseismic vs After)
  out.before_strike_deg = (pre_seis.dominant_plane.n_events >= 10)
                              ? pre_seis.dominant_plane.strike_deg
                              : geod.ridge_strike_deg;
  out.coseismic_strike_deg = inv.plane.strike_deg;
  out.after_strike_deg = (post_seis.dominant_plane.n_events >= 10)
                             ? post_seis.dominant_plane.strike_deg
                             : inv.plane.strike_deg;
  out.strike_change_deg = detail::axial_angle_diff_deg(
      out.before_strike_deg, out.coseismic_strike_deg);

  // Centroid of the cells with normalized value >= 0.75 in
  // `before_susceptibility` vs `moved_intensity`
  auto top_centroid = [&](const std::vector<double> &g) -> linalg::Vec2 {
    double sw = 0.0, cx = 0.0, cy = 0.0;
    for (std::size_t r = 0; r < grid.n_lat; ++r) {
      for (std::size_t c = 0; c < grid.n_lon; ++c) {
        const double v = g[r * grid.n_lon + c];
        if (v < 0.75)
          continue;
        const auto pt = grid.enu_at(r, c);
        sw += v;
        cx += v * pt.x;
        cy += v * pt.y;
      }
    }
    return (sw > 0.0) ? linalg::Vec2{cx / sw, cy / sw} : linalg::Vec2{};
  };

  const auto c_bef = top_centroid(out.before_susceptibility);
  const auto c_mov = top_centroid(out.moved_intensity);
  out.centroid_shift_km = (c_mov - c_bef).norm();

  return out;
}

} // namespace chaosmap::fusion
