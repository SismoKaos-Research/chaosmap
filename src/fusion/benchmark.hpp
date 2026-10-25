#pragma once

// Quantitative benchmark comparing the two Lyapunov candidates:
// - `L-geo`: Lagrangian Finite-Time Lyapunov Exponent (FTLE) of the GNSS
//   crustal velocity/deformation field
// - `L-seis`: Multi-station broadband waveform Wolf/Rosenstein Lyapunov
//   anomaly & array dispersion field
// Scores both against the coseismic Okada fault plane and aftershock rupture
// zone to determine optimal fusion weights.

#include "anomaly/array_anomaly.hpp"
#include "geo/projection.hpp"
#include "gnss/fields.hpp"
#include "okada/inversion.hpp"
#include "seis/planes.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace chaosmap::fusion {

struct CandidateScore {
  std::string name;
  bool available{false};
  double ridge_to_rupture_dist_km{999.0};
  double strike_misfit_deg{90.0};
  double spatial_roc_auc{0.5}; // Grid cell discrimination within 25 km of fault
  double temporal_pre_auc{0.5}; // Pre-mainshock horizon ROC AUC (L-seis)
  double composite_skill{0.0};  // Overall [0, 1] skill score
  double assigned_weight{0.0};
};

struct LyapunovBenchmark {
  CandidateScore l_geo;
  CandidateScore l_seis;
  std::string best_candidate;
};

namespace detail {

// Shortest distance (km) from point P to segment AB in ENU km coordinates.
[[nodiscard]] inline double point_to_segment_km(linalg::Vec2 p, linalg::Vec2 a,
                                                linalg::Vec2 b) noexcept {
  const linalg::Vec2 ab = b - a;
  const double len2 = ab.dot(ab);
  if (len2 <= 1e-9)
    return (p - a).norm();
  const double t = std::clamp((p - a).dot(ab) / len2, 0.0, 1.0);
  const linalg::Vec2 proj = a + ab * t;
  return (p - proj).norm();
}

// Acute angle [0, 90] deg between two axial strike directions (mod 180 deg).
[[nodiscard]] inline double axial_angle_diff_deg(double s1_deg,
                                                 double s2_deg) noexcept {
  double d = std::fmod(std::abs(s1_deg - s2_deg), 180.0);
  if (d > 90.0)
    d = 180.0 - d;
  return d;
}

} // namespace detail

[[nodiscard]] inline LyapunovBenchmark
benchmark_lyapunov_candidates(const gnss::GeodeticGridFields &geod,
                              const anomaly::ArrayAnomalyResult &seis_anom,
                              const okada::InversionResult &inv,
                              const geo::Grid2D &grid) {
  LyapunovBenchmark bm;
  bm.l_geo.name = "L-geo (GNSS Lagrangian FTLE)";
  bm.l_geo.available = !geod.ftle_interseismic.empty();

  bm.l_seis.name = "L-seis (Broadband Wolf/Rosenstein Lyapunov)";
  bm.l_seis.available = seis_anom.has_seismic;

  // Reference rupture segment in ENU km from inverted Okada fault plane
  const auto f_center =
      grid.proj.to_enu(inv.center_lat_deg, inv.center_lon_deg);
  const double str_rad = geo::deg2rad(inv.plane.strike_deg);
  const linalg::Vec2 s_hat{std::sin(str_rad), std::cos(str_rad)};
  const double half_L = 0.5 * inv.plane.length_km;
  const linalg::Vec2 rup_a = f_center - s_hat * half_L;
  const linalg::Vec2 rup_b = f_center + s_hat * half_L;

  // Binary label for each grid cell: 1 if within 25 km of the ruptured fault
  std::vector<std::uint8_t> near_fault(grid.size(), 0);
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const auto pt = grid.enu_at(r, c);
      const double d = detail::point_to_segment_km(pt, rup_a, rup_b);
      if (d <= 25.0)
        near_fault[r * grid.n_lon + c] = 1;
    }
  }

  // Score L-geo
  if (bm.l_geo.available) {
    const auto r0 =
        grid.proj.to_enu(geod.ridge_start_lat, geod.ridge_start_lon);
    const auto r1 = grid.proj.to_enu(geod.ridge_end_lat, geod.ridge_end_lon);
    const auto r_mid = (r0 + r1) * 0.5;
    bm.l_geo.ridge_to_rupture_dist_km =
        0.5 * (detail::point_to_segment_km(r_mid, rup_a, rup_b) +
               std::min(detail::point_to_segment_km(r0, rup_a, rup_b),
                        detail::point_to_segment_km(r1, rup_a, rup_b)));
    bm.l_geo.strike_misfit_deg = detail::axial_angle_diff_deg(
        geod.ridge_strike_deg, inv.plane.strike_deg);
    bm.l_geo.spatial_roc_auc =
        anomaly::detail::roc_auc(geod.ftle_interseismic, near_fault);
    bm.l_geo.temporal_pre_auc = 0.5; // static interseismic rate field

    const double prox_score =
        std::exp(-bm.l_geo.ridge_to_rupture_dist_km / 40.0);
    const double align_score = 1.0 - bm.l_geo.strike_misfit_deg / 90.0;
    bm.l_geo.composite_skill =
        std::clamp(0.45 * bm.l_geo.spatial_roc_auc + 0.30 * prox_score +
                       0.25 * align_score,
                   0.05, 1.0);
  }

  // Score L-seis
  if (bm.l_seis.available) {
    const auto r0 =
        grid.proj.to_enu(seis_anom.ridge_start_lat, seis_anom.ridge_start_lon);
    const auto r1 =
        grid.proj.to_enu(seis_anom.ridge_end_lat, seis_anom.ridge_end_lon);
    const auto r_mid = (r0 + r1) * 0.5;
    bm.l_seis.ridge_to_rupture_dist_km =
        0.5 * (detail::point_to_segment_km(r_mid, rup_a, rup_b) +
               std::min(detail::point_to_segment_km(r0, rup_a, rup_b),
                        detail::point_to_segment_km(r1, rup_a, rup_b)));
    bm.l_seis.strike_misfit_deg = detail::axial_angle_diff_deg(
        seis_anom.ridge_strike_deg, inv.plane.strike_deg);
    bm.l_seis.spatial_roc_auc =
        anomaly::detail::roc_auc(seis_anom.grid_l_seis_change, near_fault);
    bm.l_seis.temporal_pre_auc = seis_anom.array_pre_auc;

    const double prox_score =
        std::exp(-bm.l_seis.ridge_to_rupture_dist_km / 40.0);
    const double align_score = 1.0 - bm.l_seis.strike_misfit_deg / 90.0;
    bm.l_seis.composite_skill = std::clamp(
        0.30 * bm.l_seis.spatial_roc_auc + 0.25 * bm.l_seis.temporal_pre_auc +
            0.25 * prox_score + 0.20 * align_score,
        0.05, 1.0);
  }

  if (bm.l_geo.available && bm.l_seis.available) {
    const double s_sum = bm.l_geo.composite_skill + bm.l_seis.composite_skill;
    bm.l_geo.assigned_weight = bm.l_geo.composite_skill / s_sum;
    bm.l_seis.assigned_weight = bm.l_seis.composite_skill / s_sum;
    bm.best_candidate = (bm.l_seis.composite_skill >= bm.l_geo.composite_skill)
                            ? "Hybrid (L-seis primary + L-geo)"
                            : "Hybrid (L-geo primary + L-seis)";
  } else if (bm.l_geo.available) {
    bm.l_geo.assigned_weight = 1.0;
    bm.l_seis.assigned_weight = 0.0;
    bm.best_candidate = bm.l_geo.name;
  } else if (bm.l_seis.available) {
    bm.l_geo.assigned_weight = 0.0;
    bm.l_seis.assigned_weight = 1.0;
    bm.best_candidate = bm.l_seis.name;
  }
  return bm;
}

} // namespace chaosmap::fusion
