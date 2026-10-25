#pragma once

// GNSS trajectory modeling, MIDAS interseismic velocity estimation, and
// coseismic / postseismic offset extraction.

#include "core/linalg.hpp"
#include "geo/projection.hpp"
#include "io/tenv3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace chaosmap::gnss {

struct StationMotion {
  std::string station;
  double lat_deg{};
  double lon_deg{};
  double height_m{};
  double dist_epicenter_km{};

  // Interseismic velocity (pre-event, m/yr) in East, North, Up
  bool has_velocity{false};
  linalg::Vec3 vel_enu_m_yr{};
  linalg::Vec3 vel_sigma_m_yr{};
  int n_pre_days{0};

  // Coseismic displacement (m) in East, North, Up at `event_mjd`
  bool has_coseismic{false};
  linalg::Vec3 coseismic_enu_m{};
  linalg::Vec3 coseismic_sigma_m{};
  linalg::Vec3 wls_coseismic_enu_m{};

  // Cumulative postseismic displacement over 60 days after event (m)
  bool has_postseismic{false};
  linalg::Vec3 postseismic_60d_enu_m{};
  linalg::Vec3 postseismic_sigma_m{};
};

namespace detail {

[[nodiscard]] inline double component_val(const io::Tenv3Sample &s,
                                          int c) noexcept {
  return c == 0 ? s.east_m : (c == 1 ? s.north_m : s.up_m);
}

[[nodiscard]] inline double component_sig(const io::Tenv3Sample &s,
                                          int c) noexcept {
  return c == 0 ? s.sig_e_m : (c == 1 ? s.sig_n_m : s.sig_u_m);
}

// MIDAS-style 1-year pair median velocity on pre-event samples, skipping pairs
// that straddle known equipment steps or large (M >= 6.0) earthquake steps.
[[nodiscard]] inline std::pair<double, double>
midas_component_velocity(std::span<const io::Tenv3Sample> samples,
                         std::span<const double> step_mjds, double max_mjd,
                         int comp) {
  std::vector<double> slopes;
  std::size_t j_start = 0;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const double t1 = samples[i].mjd;
    if (t1 >= max_mjd)
      break;
    while (j_start < samples.size() && samples[j_start].mjd - t1 < 350.0)
      ++j_start;
    for (std::size_t j = j_start; j < samples.size(); ++j) {
      const double t2 = samples[j].mjd;
      if (t2 >= max_mjd || t2 - t1 > 380.0)
        break;
      // Check if any step sits in (t1, t2]
      bool crosses_step = false;
      for (double sm : step_mjds) {
        if (sm > t1 && sm <= t2) {
          crosses_step = true;
          break;
        }
      }
      if (crosses_step)
        continue;
      const double dt_yr = (t2 - t1) / 365.25;
      const double dx =
          component_val(samples[j], comp) - component_val(samples[i], comp);
      slopes.push_back(dx / dt_yr);
      break; // one ~1-yr pair per starting day, as in Blewitt et al. (2016)
    }
  }
  if (slopes.size() < 20)
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};

  const double med = linalg::median(slopes);
  const double mad = linalg::mad_sigma(slopes, med);
  // Trim 2-sigma outliers, as in MIDAS second pass
  std::vector<double> trimmed;
  trimmed.reserve(slopes.size());
  for (double v : slopes) {
    if (std::abs(v - med) <= 2.0 * mad)
      trimmed.push_back(v);
  }
  const double v_hat = trimmed.empty() ? med : linalg::median(trimmed);
  const double sig_hat =
      trimmed.empty() ? mad : linalg::mad_sigma(trimmed, v_hat);
  // Effective independent 1-yr pairs scale roughly with sqrt(N / 4) for daily
  // autocorrelated flicker noise, floored at 0.3 mm/yr horizontal, 0.8 mm/yr up.
  const double n_eff =
      std::max(1.0, static_cast<double>(trimmed.size()) / 90.0);
  const double floor_sig = (comp < 2) ? 0.0004 : 0.0010;
  const double sigma = std::hypot(sig_hat / std::sqrt(n_eff), floor_sig);
  return {v_hat, sigma};
}

} // namespace detail

// Estimates pre-event interseismic velocity, coseismic offset, and postseismic
// relaxation for a single GNSS station around `event_mjd`.
[[nodiscard]] inline StationMotion
analyze_station(const io::GnssStationSeries &series,
                std::span<const io::StepEntry> steps, double event_mjd,
                double epicenter_lat, double epicenter_lon,
                double tau_post_days = 10.0) {
  StationMotion out;
  out.station = series.station;
  out.lat_deg = series.lat_deg;
  out.lon_deg = series.lon_deg;
  out.height_m = series.height_m;
  out.dist_epicenter_km = geo::haversine_km(epicenter_lat, epicenter_lon,
                                            series.lat_deg, series.lon_deg);

  // Collect relevant step MJDs (equipment steps + M>=6.2 earthquake steps not
  // on `event_mjd`).
  std::vector<double> step_mjds;
  for (const auto &st : steps) {
    if (std::abs(st.mjd - event_mjd) <= 2.0)
      continue;
    if (st.type == 1 || (st.type == 2 && st.magnitude >= 6.2 &&
                         st.station_dist_km <= 0.5 * st.thresh_dist_km)) {
      step_mjds.push_back(st.mjd);
    }
  }
  std::ranges::sort(step_mjds);
  step_mjds.erase(std::unique(step_mjds.begin(), step_mjds.end()),
                  step_mjds.end());

  // Count pre-event samples
  for (const auto &s : series.samples) {
    if (s.mjd < event_mjd)
      ++out.n_pre_days;
  }

  // 1. Interseismic velocity via MIDAS on pre-event history (or full history
  // excluding across event_mjd if pre-event history < 400 days).
  std::vector<double> midas_steps = step_mjds;
  midas_steps.push_back(event_mjd);
  std::ranges::sort(midas_steps);
  const double vel_cutoff =
      (out.n_pre_days >= 420) ? event_mjd : (event_mjd + 5000.0);

  auto [ve, sve] =
      detail::midas_component_velocity(series.samples, midas_steps, vel_cutoff, 0);
  auto [vn, svn] =
      detail::midas_component_velocity(series.samples, midas_steps, vel_cutoff, 1);
  auto [vu, svu] =
      detail::midas_component_velocity(series.samples, midas_steps, vel_cutoff, 2);
  if (std::isfinite(ve) && std::isfinite(vn) && std::isfinite(vu)) {
    out.has_velocity = true;
    out.vel_enu_m_yr = {ve, vn, vu};
    out.vel_sigma_m_yr = {sve, svn, svu};
  }

  // 2. Coseismic offset at `event_mjd`:
  // Require at least 5 daily solutions in `[event_mjd - 30, event_mjd - 1]` and
  // at least 5 in `[event_mjd + 1, event_mjd + 30]`.
  std::vector<const io::Tenv3Sample *> pre_near, post_near, fit_win;
  for (const auto &s : series.samples) {
    if (s.mjd >= event_mjd - 30.0 && s.mjd <= event_mjd - 1.0)
      pre_near.push_back(&s);
    if (s.mjd >= event_mjd + 1.0 && s.mjd <= event_mjd + 30.0)
      post_near.push_back(&s);
    if (s.mjd >= event_mjd - 540.0 && s.mjd <= event_mjd + 365.0 &&
        std::abs(s.mjd - event_mjd) >= 0.5)
      fit_win.push_back(&s);
  }

  if (pre_near.size() < 4 || post_near.size() < 4)
    return out;

  // Identify equipment steps that genuinely sit inside the WLS fit window
  // with at least 10 samples on both sides.
  std::vector<double> win_equip_steps;
  for (double sm : step_mjds) {
    int n_before = 0, n_after = 0;
    for (const auto *s : fit_win) {
      if (s->mjd < sm)
        ++n_before;
      else
        ++n_after;
    }
    if (n_before >= 10 && n_after >= 10)
      win_equip_steps.push_back(sm);
  }

  // Check if we have enough post-event span (>45 days) to separate postseismic
  // log relaxation from the coseismic step.
  int n_post_long = 0;
  for (const auto *s : fit_win) {
    if (s->mjd > event_mjd + 30.0)
      ++n_post_long;
  }
  const bool fit_post_log = (n_post_long >= 15);
  const bool fit_seasonal = (fit_win.size() >= 250);

  double co_vals[3]{};
  double co_sigs[3]{};
  double wls_co_vals[3]{};
  double post_vals[3]{};
  double post_sigs[3]{};

  for (int comp = 0; comp < 3; ++comp) {
    const double v_sec =
        out.has_velocity
            ? (comp == 0 ? out.vel_enu_m_yr.x
                         : (comp == 1 ? out.vel_enu_m_yr.y : out.vel_enu_m_yr.z))
            : 0.0;

    // A) Robust detrended pre/post median difference over [-21d, -1d] vs [+1d, +15d]
    std::vector<double> pre_v, post_v;
    for (const auto *s : pre_near) {
      if (s->mjd >= event_mjd - 21.0) {
        const double dt_yr = (s->mjd - event_mjd) / 365.25;
        pre_v.push_back(detail::component_val(*s, comp) - v_sec * dt_yr);
      }
    }
    for (const auto *s : post_near) {
      if (s->mjd <= event_mjd + 16.0) {
        const double dt_yr = (s->mjd - event_mjd) / 365.25;
        post_v.push_back(detail::component_val(*s, comp) - v_sec * dt_yr);
      }
    }
    if (pre_v.size() < 4) {
      for (const auto *s : pre_near) {
        const double dt_yr = (s->mjd - event_mjd) / 365.25;
        pre_v.push_back(detail::component_val(*s, comp) - v_sec * dt_yr);
      }
    }
    if (post_v.size() < 4) {
      for (const auto *s : post_near) {
        const double dt_yr = (s->mjd - event_mjd) / 365.25;
        post_v.push_back(detail::component_val(*s, comp) - v_sec * dt_yr);
      }
    }
    const double med_pre = linalg::median(pre_v);
    const double med_post = linalg::median(post_v);
    const double sig_pre = linalg::mad_sigma(pre_v, med_pre);
    const double sig_post = linalg::mad_sigma(post_v, med_post);
    const double delta_med = med_post - med_pre;
    const double floor_mm = (comp < 2) ? 0.0012 : 0.0035;
    const double sigma_med = std::hypot(
        std::hypot(sig_pre / std::sqrt(static_cast<double>(pre_v.size())),
                   sig_post / std::sqrt(static_cast<double>(post_v.size()))),
        floor_mm);

    // B) Weighted Least-Squares trajectory model over `fit_win`
    // Columns: 0: intercept, 1: velocity, 2: coseismic H(t - t_eq),
    // optional: postseismic log, seasonal (4 cols), equipment steps.
    std::size_t col_co = 2;
    std::size_t col_post = fit_post_log ? 3 : 0;
    std::size_t col_next = fit_post_log ? 4 : 3;
    std::size_t col_seas = fit_seasonal ? col_next : 0;
    if (fit_seasonal)
      col_next += 4;
    std::size_t col_eq_base = col_next;
    col_next += win_equip_steps.size();

    const std::size_t m = fit_win.size();
    linalg::Mat A(m, col_next, 0.0);
    std::vector<double> b(m, 0.0);
    std::vector<double> sig(m, 1.0);

    for (std::size_t i = 0; i < m; ++i) {
      const auto *s = fit_win[i];
      const double dt_days = s->mjd - event_mjd;
      const double tau_yr = dt_days / 365.25;
      A(i, 0) = 1.0;
      A(i, 1) = tau_yr;
      if (dt_days > 0.0) {
        A(i, col_co) = 1.0;
        if (fit_post_log)
          A(i, col_post) = std::log(1.0 + dt_days / tau_post_days);
      }
      if (fit_seasonal) {
        const double w1 = 2.0 * std::numbers::pi * tau_yr;
        const double w2 = 4.0 * std::numbers::pi * tau_yr;
        A(i, col_seas + 0) = std::cos(w1);
        A(i, col_seas + 1) = std::sin(w1);
        A(i, col_seas + 2) = std::cos(w2);
        A(i, col_seas + 3) = std::sin(w2);
      }
      for (std::size_t k = 0; k < win_equip_steps.size(); ++k) {
        if (s->mjd >= win_equip_steps[k])
          A(i, col_eq_base + k) = 1.0;
      }
      b[i] = detail::component_val(*s, comp);
      sig[i] = std::max(1e-4, detail::component_sig(*s, comp));
    }

    const auto wls = linalg::solve_wls(A, b, sig);
    const double rrms =
        (wls.dof > 0) ? std::sqrt(wls.rss / static_cast<double>(wls.dof)) : 1.0;
    wls_co_vals[comp] = wls.x[col_co];

    // Use the robust local detrended jump as primary coseismic offset (less
    // collinear with log postseismic decay on day 1-15), and combine its
    // uncertainty with the WLS formal uncertainty.
    co_vals[comp] = delta_med;
    const double wls_sig =
        std::sqrt(std::max(0.0, wls.cov(col_co, col_co))) * rrms;
    co_sigs[comp] = std::max(sigma_med, 0.5 * wls_sig);

    if (fit_post_log) {
      const double log_60 = std::log(1.0 + 60.0 / tau_post_days);
      post_vals[comp] = wls.x[col_post] * log_60;
      post_sigs[comp] =
          std::sqrt(std::max(0.0, wls.cov(col_post, col_post))) * rrms * log_60;
    }
  }

  out.has_coseismic = true;
  out.coseismic_enu_m = {co_vals[0], co_vals[1], co_vals[2]};
  out.coseismic_sigma_m = {co_sigs[0], co_sigs[1], co_sigs[2]};
  out.wls_coseismic_enu_m = {wls_co_vals[0], wls_co_vals[1], wls_co_vals[2]};
  if (fit_post_log) {
    out.has_postseismic = true;
    out.postseismic_60d_enu_m = {post_vals[0], post_vals[1], post_vals[2]};
    out.postseismic_sigma_m = {post_sigs[0], post_sigs[1], post_sigs[2]};
  }
  return out;
}

} // namespace chaosmap::gnss
