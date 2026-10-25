#pragma once

// Fault-plane geometry search and regularized distributed slip inversion using
// the Okada (1985) half-space Green's functions.

#include "core/linalg.hpp"
#include "geo/projection.hpp"
#include "gnss/trajectory.hpp"
#include "okada/okada85.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

namespace chaosmap::okada {

struct SubPatchResult {
  FaultPatch patch;
  double lat_deg{};
  double lon_deg{};
  double total_slip_m{};
  double rake_deg{};
  double resolution{}; // diagonal of model resolution matrix R_ii in [0, 1]
};

struct InversionResult {
  FaultPatch plane; // overall best-fit single plane
  double center_lat_deg{};
  double center_lon_deg{};
  double trace_start_lat{};
  double trace_start_lon{};
  double trace_end_lat{};
  double trace_end_lon{};
  double mean_slip_m{};
  double max_slip_m{};
  double mean_rake_deg{};
  double moment_nm{};
  double mw_geodetic{};
  double wrms_mm{};
  double variance_reduction_pct{};
  double smoothing_lambda{};
  std::vector<SubPatchResult> patches;
  std::vector<linalg::Vec3> predicted_enu_m; // per input coseismic station
};

namespace detail {

// Solves 2-parameter (U1, U2) weighted least squares for a single candidate
// `FaultPatch` against station coseismic offsets, regularized weakly toward
// the reference rake and seismic moment when the network is one-sided.
[[nodiscard]] inline std::pair<linalg::Vec2, double> fit_single_patch_slip(
    FaultPatch p, std::span<const gnss::StationMotion> stations,
    const geo::LocalProjection &proj, double rake_hint_deg, double s_prior_m) {
  const std::size_t n_sta = stations.size();
  const std::size_t m_rows = 3 * n_sta + 2;
  linalg::Mat G(m_rows, 2, 0.0);
  std::vector<double> d(m_rows, 0.0);

  for (std::size_t i = 0; i < n_sta; ++i) {
    const auto &st = stations[i];
    const auto obs = proj.to_enu(st.lat_deg, st.lon_deg);

    p.slip_strike_m = 1.0;
    p.slip_dip_m = 0.0;
    const auto g1 = dislocation_enu(p, obs.x, obs.y);

    p.slip_strike_m = 0.0;
    p.slip_dip_m = 1.0;
    const auto g2 = dislocation_enu(p, obs.x, obs.y);

    const double we = 1.0 / std::max(0.002, st.coseismic_sigma_m.x);
    const double wn = 1.0 / std::max(0.002, st.coseismic_sigma_m.y);
    // Downweight vertical component by 0.5x beyond its formal sigma
    const double wu = 0.5 / std::max(0.006, st.coseismic_sigma_m.z);

    G(3 * i + 0, 0) = g1.x * we;
    G(3 * i + 0, 1) = g2.x * we;
    d[3 * i + 0] = st.coseismic_enu_m.x * we;

    G(3 * i + 1, 0) = g1.y * wn;
    G(3 * i + 1, 1) = g2.y * wn;
    d[3 * i + 1] = st.coseismic_enu_m.y * wn;

    G(3 * i + 2, 0) = g1.z * wu;
    G(3 * i + 2, 1) = g2.z * wu;
    d[3 * i + 2] = st.coseismic_enu_m.z * wu;
  }

  // Weak mechanism damping so underconstrained/one-sided geometries stay
  // physically consistent with the reference mechanism.
  const double rake_rad = geo::deg2rad(rake_hint_deg);
  const double w_prior = 0.45 / std::max(0.2, s_prior_m);
  G(3 * n_sta + 0, 0) = w_prior;
  d[3 * n_sta + 0] = s_prior_m * std::cos(rake_rad) * w_prior;
  G(3 * n_sta + 1, 1) = w_prior;
  d[3 * n_sta + 1] = s_prior_m * std::sin(rake_rad) * w_prior;

  const auto u = linalg::solve_lstsq(G, d);
  const auto pred = G.mul_vec(u);
  double cost = 0.0;
  for (std::size_t r = 0; r < m_rows; ++r) {
    const double diff = d[r] - pred[r];
    cost += diff * diff;
  }
  return {{u[0], u[1]}, cost};
}

} // namespace detail

// Runs two-stage fault inversion:
// Stage 1: Nonlinear search over fault centroid, strike, dip, depth, length,
//          width around the event prior.
// Stage 2: Subdivides the optimal plane into `n_strike x n_dip` patches and
//          solves the Laplacian-smoothed distributed slip problem + resolution.
[[nodiscard]] inline InversionResult
invert_fault(std::span<const gnss::StationMotion> all_stations,
             const geo::LocalProjection &proj, double mw_ref,
             double depth_hint_km, double strike_hint_deg, double dip_hint_deg,
             double rake_hint_deg, bool is_strike_slip,
             std::size_t n_strike = 8, std::size_t n_dip = 4,
             double shear_modulus_pa = 30.0e9) {
  std::vector<gnss::StationMotion> stations;
  for (const auto &s : all_stations) {
    if (s.has_coseismic)
      stations.push_back(s);
  }

  InversionResult out;
  if (stations.empty())
    return out;

  // Empirical Wells & Coppersmith (1994) initial dimensions from mw_ref:
  const double m0_ref = std::pow(10.0, 1.5 * mw_ref + 9.1);
  const double l0_km =
      std::clamp(std::pow(10.0, -2.44 + 0.59 * mw_ref), 15.0, 260.0);
  const double w0_km =
      std::clamp(std::pow(10.0, -1.01 + 0.32 * mw_ref), 10.0, 35.0);
  const double s_prior_m =
      m0_ref / (shear_modulus_pa * (l0_km * 1e3) * (w0_km * 1e3));

  FaultPatch best;
  best.east_km = 0.0;
  best.north_km = 0.0;
  best.depth_km = std::max(6.0, depth_hint_km);
  best.length_km = l0_km;
  best.width_km = w0_km;
  best.strike_deg = strike_hint_deg;
  best.dip_deg = dip_hint_deg;

  double best_cost = 1e30;
  for (double de : {-18.0, -9.0, 0.0, 9.0, 18.0}) {
    for (double dn : {-18.0, -9.0, 0.0, 9.0, 18.0}) {
      for (double ds : {-16.0, -8.0, 0.0, 8.0, 16.0}) {
        for (double dd : {-10.0, 0.0, 10.0}) {
          for (double l_scale : {0.8, 1.0, 1.25}) {
            FaultPatch cand = best;
            cand.east_km = de;
            cand.north_km = dn;
            cand.strike_deg = strike_hint_deg + ds;
            cand.dip_deg = std::clamp(dip_hint_deg + dd, 15.0, 89.0);
            cand.length_km = l0_km * l_scale;
            cand.width_km = w0_km;
            cand.depth_km = std::max(0.55 * cand.width_km *
                                         std::sin(geo::deg2rad(cand.dip_deg)),
                                     depth_hint_km);
            auto [sl, cost] = detail::fit_single_patch_slip(
                cand, stations, proj, rake_hint_deg, s_prior_m);
            // Penalize centroid offset weakly so poorly constrained directions
            // stay anchored near the epicentre
            cost += 0.015 * (de * de + dn * dn) + 0.01 * ds * ds;
            if (cost < best_cost) {
              best_cost = cost;
              cand.slip_strike_m = sl.x;
              cand.slip_dip_m = sl.y;
              best = cand;
            }
          }
        }
      }
    }
  }

  // Coordinate-pattern polish around `best`
  for (double step : {4.0, 1.5}) {
    bool improved = true;
    while (improved) {
      improved = false;
      for (int param = 0; param < 4; ++param) {
        for (double dir : {-1.0, 1.0}) {
          FaultPatch cand = best;
          if (param == 0)
            cand.east_km += dir * step;
          else if (param == 1)
            cand.north_km += dir * step;
          else if (param == 2)
            cand.strike_deg += dir * step;
          else
            cand.dip_deg = std::clamp(cand.dip_deg + dir * step, 15.0, 89.0);

          auto [sl, cost] = detail::fit_single_patch_slip(
              cand, stations, proj, rake_hint_deg, s_prior_m);
          const double dstr = cand.strike_deg - strike_hint_deg;
          cost += 0.015 * (cand.east_km * cand.east_km +
                           cand.north_km * cand.north_km) +
                  0.01 * dstr * dstr;
          if (cost + 1e-6 < best_cost) {
            best_cost = cost;
            cand.slip_strike_m = sl.x;
            cand.slip_dip_m = sl.y;
            best = cand;
            improved = true;
          }
        }
      }
    }
  }

  out.plane = best;
  const auto center_ll = proj.to_latlon(best.east_km, best.north_km);
  out.center_lat_deg = center_ll.x;
  out.center_lon_deg = center_ll.y;

  // Up-dip surface projection of the fault top edge (fault trace):
  const double str_rad = geo::deg2rad(best.strike_deg);
  const double dip_rad = geo::deg2rad(best.dip_deg);
  const double sin_s = std::sin(str_rad);
  const double cos_s = std::cos(str_rad);
  const double up_dip_horiz = 0.5 * best.width_km * std::cos(dip_rad);
  // Up-dip direction is to the LEFT of strike: (-cos_s, sin_s)
  const double top_e = best.east_km - up_dip_horiz * cos_s;
  const double top_n = best.north_km + up_dip_horiz * sin_s;
  const auto tr_start = proj.to_latlon(top_e - 0.5 * best.length_km * sin_s,
                                       top_n - 0.5 * best.length_km * cos_s);
  const auto tr_end = proj.to_latlon(top_e + 0.5 * best.length_km * sin_s,
                                     top_n + 0.5 * best.length_km * cos_s);
  out.trace_start_lat = tr_start.x;
  out.trace_start_lon = tr_start.y;
  out.trace_end_lat = tr_end.x;
  out.trace_end_lon = tr_end.y;

  // Stage 2: Distributed slip on `n_strike x n_dip` patches
  const std::size_t n_patches = n_strike * n_dip;
  const double dl = best.length_km / static_cast<double>(n_strike);
  const double dw = best.width_km / static_cast<double>(n_dip);

  std::vector<FaultPatch> grid_patches(n_patches);
  for (std::size_t ir = 0; ir < n_dip; ++ir) {
    // ir = 0 is shallowest row, ir = n_dip - 1 is deepest row
    const double v_off =
        (static_cast<double>(ir) + 0.5 - 0.5 * static_cast<double>(n_dip)) * dw;
    for (std::size_t ic = 0; ic < n_strike; ++ic) {
      const double u_off = (static_cast<double>(ic) + 0.5 -
                            0.5 * static_cast<double>(n_strike)) *
                           dl;
      // u_off is along-strike; v_off > 0 is down-dip (to the right of strike:
      // (+cos_s, -sin_s) horizontally, +sin(dip) in depth)
      FaultPatch sub = best;
      sub.length_km = dl;
      sub.width_km = dw;
      sub.east_km =
          best.east_km + u_off * sin_s + v_off * std::cos(dip_rad) * cos_s;
      sub.north_km =
          best.north_km + u_off * cos_s - v_off * std::cos(dip_rad) * sin_s;
      sub.depth_km = std::max(0.5 * dw * std::sin(dip_rad) + 0.2,
                              best.depth_km + v_off * std::sin(dip_rad));
      grid_patches[ir * n_strike + ic] = sub;
    }
  }

  // Build weighted data Green's matrix G_d (3*n_sta x 2*n_patches)
  const std::size_t n_sta = stations.size();
  const std::size_t n_obs = 3 * n_sta;
  const std::size_t n_params = 2 * n_patches;
  linalg::Mat Gd(n_obs, n_params, 0.0);
  std::vector<double> d_w(n_obs, 0.0);

  for (std::size_t i = 0; i < n_sta; ++i) {
    const auto &st = stations[i];
    const auto obs = proj.to_enu(st.lat_deg, st.lon_deg);
    const double we = 1.0 / std::max(0.002, st.coseismic_sigma_m.x);
    const double wn = 1.0 / std::max(0.002, st.coseismic_sigma_m.y);
    const double wu = 0.5 / std::max(0.006, st.coseismic_sigma_m.z);

    d_w[3 * i + 0] = st.coseismic_enu_m.x * we;
    d_w[3 * i + 1] = st.coseismic_enu_m.y * wn;
    d_w[3 * i + 2] = st.coseismic_enu_m.z * wu;

    for (std::size_t j = 0; j < n_patches; ++j) {
      FaultPatch p1 = grid_patches[j];
      p1.slip_strike_m = 1.0;
      p1.slip_dip_m = 0.0;
      const auto g1 = dislocation_enu(p1, obs.x, obs.y);

      p1.slip_strike_m = 0.0;
      p1.slip_dip_m = 1.0;
      const auto g2 = dislocation_enu(p1, obs.x, obs.y);

      Gd(3 * i + 0, 2 * j + 0) = g1.x * we;
      Gd(3 * i + 0, 2 * j + 1) = g2.x * we;
      Gd(3 * i + 1, 2 * j + 0) = g1.y * wn;
      Gd(3 * i + 1, 2 * j + 1) = g2.y * wn;
      Gd(3 * i + 2, 2 * j + 0) = g1.z * wu;
      Gd(3 * i + 2, 2 * j + 1) = g2.z * wu;
    }
  }

  // Build 2D Laplacian + weak prior matrix
  // Choose smoothing lambda scaled to ||Gd||_F / sqrt(n_params)
  double gd_norm2 = 0.0;
  for (std::size_t c = 0; c < n_params; ++c)
    for (std::size_t r = 0; r < n_obs; ++r)
      gd_norm2 += Gd(r, c) * Gd(r, c);
  const double lambda =
      std::max(0.5, 0.35 * std::sqrt(gd_norm2 / static_cast<double>(n_params)));
  out.smoothing_lambda = lambda;

  const std::size_t total_rows = n_obs + 2 * n_params;
  linalg::Mat A_aug(total_rows, n_params, 0.0);
  std::vector<double> b_aug(total_rows, 0.0);
  for (std::size_t r = 0; r < n_obs; ++r) {
    b_aug[r] = d_w[r];
    for (std::size_t c = 0; c < n_params; ++c)
      A_aug(r, c) = Gd(r, c);
  }

  // Discrete 2D Laplacian over patches
  for (std::size_t ir = 0; ir < n_dip; ++ir) {
    for (std::size_t ic = 0; ic < n_strike; ++ic) {
      const std::size_t j = ir * n_strike + ic;
      for (std::size_t comp = 0; comp < 2; ++comp) {
        const std::size_t row = n_obs + 2 * j + comp;
        int deg = 0;
        if (ir > 0) {
          A_aug(row, 2 * ((ir - 1) * n_strike + ic) + comp) = -lambda;
          ++deg;
        }
        if (ir + 1 < n_dip) {
          A_aug(row, 2 * ((ir + 1) * n_strike + ic) + comp) = -lambda;
          ++deg;
        }
        if (ic > 0) {
          A_aug(row, 2 * (ir * n_strike + (ic - 1)) + comp) = -lambda;
          ++deg;
        }
        if (ic + 1 < n_strike) {
          A_aug(row, 2 * (ir * n_strike + (ic + 1)) + comp) = -lambda;
          ++deg;
        }
        A_aug(row, 2 * j + comp) = lambda * (static_cast<double>(deg) + 0.15);
      }
    }
  }

  // Tapered prior toward single-plane slip `best` so unilluminated patches
  // smoothly decay rather than oscillating
  const double w_ref = 0.45 * lambda;
  for (std::size_t ir = 0; ir < n_dip; ++ir) {
    for (std::size_t ic = 0; ic < n_strike; ++ic) {
      const std::size_t j = ir * n_strike + ic;
      // Cosine taper along strike and depth
      const double tx =
          std::sin(std::numbers::pi * (static_cast<double>(ic) + 0.5) /
                   static_cast<double>(n_strike));
      const double ty =
          std::sin(std::numbers::pi * (static_cast<double>(ir) + 0.5) /
                   static_cast<double>(n_dip));
      const double taper = 1.25 * tx * ty;
      const std::size_t row0 = n_obs + n_params + 2 * j + 0;
      const std::size_t row1 = n_obs + n_params + 2 * j + 1;
      A_aug(row0, 2 * j + 0) = w_ref;
      b_aug[row0] = w_ref * best.slip_strike_m * taper;
      A_aug(row1, 2 * j + 1) = w_ref;
      b_aug[row1] = w_ref * best.slip_dip_m * taper;
      if (is_strike_slip) {
        // Penalize large spurious dip-slip on predominantly strike-slip faults
        A_aug(row1, 2 * j + 1) = 2.0 * w_ref;
      }
    }
  }

  const auto sol = linalg::solve_lstsq(A_aug, b_aug);

  // Compute patch resolution proxy from column norms of Gd vs regularization
  out.patches.resize(n_patches);
  double sum_slip = 0.0;
  double max_slip = 0.0;
  double sum_u1 = 0.0, sum_u2 = 0.0;
  const double patch_area_m2 = (dl * 1e3) * (dw * 1e3);

  for (std::size_t j = 0; j < n_patches; ++j) {
    FaultPatch p = grid_patches[j];
    p.slip_strike_m = sol[2 * j + 0];
    p.slip_dip_m = sol[2 * j + 1];
    const double s = std::hypot(p.slip_strike_m, p.slip_dip_m);
    const double rake = geo::rad2deg(std::atan2(p.slip_dip_m, p.slip_strike_m));
    const auto ll = proj.to_latlon(p.east_km, p.north_km);

    double g_col2 = 0.0;
    for (std::size_t r = 0; r < n_obs; ++r) {
      g_col2 += Gd(r, 2 * j + 0) * Gd(r, 2 * j + 0) +
                Gd(r, 2 * j + 1) * Gd(r, 2 * j + 1);
    }
    const double res = g_col2 / (g_col2 + lambda * lambda);

    out.patches[j] = SubPatchResult{
        .patch = p,
        .lat_deg = ll.x,
        .lon_deg = ll.y,
        .total_slip_m = s,
        .rake_deg = rake,
        .resolution = res,
    };
    sum_slip += s;
    max_slip = std::max(max_slip, s);
    sum_u1 += p.slip_strike_m;
    sum_u2 += p.slip_dip_m;
  }

  out.mean_slip_m = sum_slip / static_cast<double>(n_patches);
  out.max_slip_m = max_slip;
  out.mean_rake_deg = geo::rad2deg(std::atan2(sum_u2, sum_u1));
  out.moment_nm = shear_modulus_pa * patch_area_m2 * sum_slip;
  out.mw_geodetic = (out.moment_nm > 0.0)
                        ? (2.0 / 3.0) * (std::log10(out.moment_nm) - 9.1)
                        : 0.0;

  // Predicted displacements at each station & variance reduction
  out.predicted_enu_m.resize(n_sta);
  double rss_m2 = 0.0, tss_m2 = 0.0;
  for (std::size_t i = 0; i < n_sta; ++i) {
    const auto &st = stations[i];
    const auto obs = proj.to_enu(st.lat_deg, st.lon_deg);
    linalg::Vec3 pred{};
    for (const auto &sp : out.patches)
      pred = pred + dislocation_enu(sp.patch, obs.x, obs.y);
    out.predicted_enu_m[i] = pred;
    const auto diff = st.coseismic_enu_m - pred;
    rss_m2 += diff.x * diff.x + diff.y * diff.y + 0.25 * diff.z * diff.z;
    tss_m2 += st.coseismic_enu_m.x * st.coseismic_enu_m.x +
              st.coseismic_enu_m.y * st.coseismic_enu_m.y +
              0.25 * st.coseismic_enu_m.z * st.coseismic_enu_m.z;
  }
  out.wrms_mm =
      1000.0 * std::sqrt(rss_m2 / static_cast<double>(
                                      std::max<std::size_t>(1, 3 * n_sta)));
  out.variance_reduction_pct =
      (tss_m2 > 0.0) ? 100.0 * (1.0 - rss_m2 / tss_m2) : 0.0;
  return out;
}

} // namespace chaosmap::okada
