#pragma once

// Multi-station broadband Lyapunov (`L-seis`) pipeline:
// - Event-coda masking (Lee 1972 duration formula) so only background windows
//   are analyzed
// - Dual-band extraction: FaultBand [0.05, 0.5 Hz] vs ControlBand [0.5, 1.8 Hz]
// - Trailing robust z_IQR per station + cross-band difference + inter-station
//   array dispersion (`array_disp`)
// - Pre-event vs post-event state comparison and circular time-slide ROC AUC
// - Spatial interpolation onto Grid2D (`grid_l_seis_pre`, `grid_l_seis_post`,
//   `grid_l_seis_change`).

#include "chaos/lyapunov.hpp"
#include "core/config.hpp"
#include "core/linalg.hpp"
#include "dsp/filter.hpp"
#include "geo/projection.hpp"
#include "io/catalog.hpp"
#include "io/zip_mseed.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace chaosmap::anomaly {

struct DailyStationLyapunov {
  // MJD on which this 24 h bin starts; bins are aligned to the mainshock origin
  // time, so day_mjd < floor(origin MJD) means entirely before the mainshock.
  int day_mjd{};
  double wolf_fault{};
  double wolf_ctrl{};
  double ros_short_fault{};
  double samp_ent_fault{};
  double corr_dim_fault{};
  double z_iqr_wolf{};
  double z_iqr_crossband{};
  std::size_t n_windows{0};
};

struct SeismicStationSummary {
  std::string station;
  double lat_deg{};
  double lon_deg{};
  double dist_to_hypo_km{};
  double pre_wolf_median{};
  double post_wolf_median{};
  double delta_wolf_post_minus_pre{};
  double pre_ros_short_median{};
  double post_ros_short_median{};
  double delta_ros_short{};
  double pre_horizon_max_z{};       // max robust z in last 15 days before event
  double pre_horizon_crossband_z{}; // cross-band anomaly in last 15 days
  double station_auc{0.5};
  std::vector<DailyStationLyapunov> daily;
};

struct DailyArrayDispersion {
  int day_mjd{};
  double mean_z{};
  double spatial_std_z{}; // inter-station dispersion (`array_disp`)
  double max_z{};
  std::size_t n_active_stations{0};
};

struct ArrayAnomalyResult {
  bool has_seismic{false};
  std::vector<SeismicStationSummary> stations;
  std::vector<DailyArrayDispersion> array_daily;
  double array_pre_auc{0.5};
  double array_null_p_value{1.0};
  double mean_abs_delta_wolf{};

  // Spatial grids on Grid2D (normalized [0, 1])
  std::vector<double> grid_l_seis_pre;
  std::vector<double> grid_l_seis_post;
  std::vector<double> grid_l_seis_change;

  // Principal axis of L-seis anomaly ridge:
  double ridge_start_lat{};
  double ridge_start_lon{};
  double ridge_end_lat{};
  double ridge_end_lon{};
  double ridge_strike_deg{};
};

namespace detail {

// Signal duration in days for an earthquake of magnitude M, from the Lee et
// al. (1972) duration magnitude M = -0.87 + 2.00 log10(tau) + 0.0035 D solved
// for tau with the distance term dropped (D = 0 gives the longest duration):
// tau = 10^((M + 0.87) / 2) s, plus a conservative 120 s buffer, capped at 2 h.
[[nodiscard]] inline double coda_duration_days(double mag) noexcept {
  if (mag < 2.2)
    return 0.0;
  const double sec =
      std::clamp(std::pow(10.0, 0.5 * (mag + 0.87)) + 120.0, 120.0, 7200.0);
  return sec / 86400.0;
}

[[nodiscard]] inline bool
window_overlaps_catalog(double w_start_mjd, double w_end_mjd,
                        std::span<const io::CatalogEvent> events) noexcept {
  for (const auto &ev : events) {
    if (ev.magnitude < 4.3)
      continue;
    const double ev_mjd = 40587.0 + static_cast<double>(ev.epoch_s) / 86400.0;
    const double ev_end = ev_mjd + coda_duration_days(ev.magnitude);
    if (!(w_end_mjd < ev_mjd || w_start_mjd > ev_end))
      return true;
  }
  return false;
}

// Robust z_IQR = (x - median(ref)) / (IQR(ref) / 1.349)
[[nodiscard]] inline std::vector<double>
compute_robust_z(std::span<const double> x, std::span<const double> ref) {
  std::vector<double> r_vec(ref.begin(), ref.end());
  std::erase_if(r_vec, [](double v) { return !std::isfinite(v); });
  std::vector<double> out(x.size(), 0.0);
  if (r_vec.size() < 3)
    return out;

  const double med = linalg::median(r_vec);
  std::vector<double> q25_v = r_vec;
  std::vector<double> q75_v = r_vec;
  const double q25 = linalg::quantile(q25_v, 0.25);
  const double q75 = linalg::quantile(q75_v, 0.75);
  const double scale = std::max(1e-6, (q75 - q25) / 1.349);

  for (std::size_t i = 0; i < x.size(); ++i) {
    if (std::isfinite(x[i]))
      out[i] = (x[i] - med) / scale;
  }
  return out;
}

// Mann-Whitney / ROC AUC for binary labels (1 = pre-event horizon, 0 =
// background)
[[nodiscard]] inline double
roc_auc(std::span<const double> scores,
        std::span<const std::uint8_t> is_horizon) noexcept {
  std::size_t n_pos = 0, n_neg = 0;
  for (auto b : is_horizon) {
    if (b)
      ++n_pos;
    else
      ++n_neg;
  }
  if (n_pos == 0 || n_neg == 0)
    return 0.5;

  double wins = 0.0;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (!is_horizon[i])
      continue;
    for (std::size_t j = 0; j < scores.size(); ++j) {
      if (is_horizon[j])
        continue;
      if (scores[i] > scores[j])
        wins += 1.0;
      else if (scores[i] == scores[j])
        wins += 0.5;
    }
  }
  return wins / static_cast<double>(n_pos * n_neg);
}

} // namespace detail

[[nodiscard]] inline ArrayAnomalyResult
analyze_broadband_array(const core::CaseConfig &cfg,
                        std::span<const io::CatalogEvent> events,
                        const geo::Grid2D &grid) {
  ArrayAnomalyResult out;
  out.grid_l_seis_pre.assign(grid.size(), 0.0);
  out.grid_l_seis_post.assign(grid.size(), 0.0);
  out.grid_l_seis_change.assign(grid.size(), 0.0);

  if (cfg.paths.broadband_dir.empty())
    return out;
  std::error_code ec;
  if (!std::filesystem::exists(cfg.paths.broadband_dir, ec))
    return out;

  const auto coords = io::read_station_coords_csv(
      cfg.paths.station_coords_csv, cfg.paths.afad_station_catalog_csv);

  std::vector<std::filesystem::path> sta_dirs;
  for (const auto &entry :
       std::filesystem::directory_iterator(cfg.paths.broadband_dir, ec)) {
    if (entry.is_directory())
      sta_dirs.push_back(entry.path());
  }
  std::ranges::sort(sta_dirs);
  if (sta_dirs.empty())
    return out;

  constexpr double kWinSec = 600.0;
  constexpr double kTargetFs = 5.0;
  constexpr double kFmin = 0.05;
  constexpr double kFmax = 0.50;
  constexpr int kTau = 5;
  constexpr int kDim = 5;
  constexpr int kEvolve = 5;

  // Mainshock origin time as a fractional MJD (`event.mjd` is only the day)
  const double origin_mjd =
      (cfg.event.origin_epoch_s != 0)
          ? 40587.0 + static_cast<double>(cfg.event.origin_epoch_s) / 86400.0
          : cfg.event.mjd;
  const int ev_day = static_cast<int>(std::floor(origin_mjd));

  // Analyze [origin - 8 d, origin + 7 d] across all stations in parallel
  const double min_mjd = origin_mjd - 8.0;
  const double max_mjd = origin_mjd + 7.0;
  const double win_days = kWinSec / 86400.0;

  std::vector<SeismicStationSummary> summaries(sta_dirs.size());
  std::vector<std::thread> workers;
  workers.reserve(sta_dirs.size());

  for (std::size_t s_idx = 0; s_idx < sta_dirs.size(); ++s_idx) {
    workers.emplace_back([&, s_idx]() {
      const auto &dir = sta_dirs[s_idx];
      const std::string sta_name = dir.filename().string();
      SeismicStationSummary sum;
      sum.station = sta_name;

      for (const auto &sc : coords) {
        if (sc.station == sta_name) {
          sum.lat_deg = sc.lat_deg;
          sum.lon_deg = sc.lon_deg;
          break;
        }
      }
      if (sum.lat_deg == 0.0 && sum.lon_deg == 0.0)
        return;

      sum.dist_to_hypo_km = geo::haversine_km(sum.lat_deg, sum.lon_deg,
                                              cfg.event.lat, cfg.event.lon);

      auto traces = io::load_station_waveforms(dir, min_mjd, max_mjd, "BH");
      if (traces.empty())
        return;

      struct DayAccum {
        std::vector<double> wolf_f;
        std::vector<double> wolf_c;
        std::vector<double> ros_f;
        std::vector<double> se_f;
        std::vector<double> cd_f;
      };
      std::map<int, DayAccum> by_day;

      for (const auto &tr : traces) {
        if (tr.sample_rate_hz < 10.0)
          continue;
        const auto win_n =
            static_cast<std::size_t>(std::llround(kWinSec * tr.sample_rate_hz));
        // Sample one 10-minute background window every 3 hours (stride of 18
        // windows, i.e. 8 windows per day)
        const std::size_t stride_n = win_n * 18;
        if (tr.data.size() < win_n)
          continue;

        for (std::size_t off = 0; off + win_n <= tr.data.size();
             off += stride_n) {
          const double w_mjd =
              tr.start_mjd +
              (static_cast<double>(off) / tr.sample_rate_hz) / 86400.0;
          if (w_mjd < min_mjd || w_mjd > max_mjd)
            continue;
          // Exclude windows overlapping [origin - 1.2 h, origin + 8.4 h]
          if (w_mjd + win_days >= origin_mjd - 0.05 &&
              w_mjd <= origin_mjd + 0.35)
            continue;
          if (detail::window_overlaps_catalog(w_mjd, w_mjd + win_days, events))
            continue;

          const std::span<const double> raw_span(tr.data.data() + off, win_n);
          const auto filt_fault = dsp::preprocess_window(
              raw_span, tr.sample_rate_hz, kTargetFs, kFmin, kFmax);
          const auto wf_fault = chaos::extract_window_features(
              filt_fault, kTargetFs, w_mjd, kTau, kDim, kEvolve);

          // Control band [0.5, 1.8 Hz] (secondary microseism / cultural)
          const auto filt_ctrl = dsp::preprocess_window(
              raw_span, tr.sample_rate_hz, kTargetFs, 0.5, 1.8);
          const auto x_ctrl_norm = chaos::zscore_normalize(filt_ctrl);
          const double wolf_ctrl =
              chaos::wolf_lye_core(x_ctrl_norm, kTargetFs, kTau, kDim, kEvolve);

          if (std::isfinite(wf_fault.wolf_lye)) {
            // 24 h bins aligned to the origin time (see DailyStationLyapunov)
            const int day =
                ev_day + static_cast<int>(std::floor(w_mjd - origin_mjd));
            auto &acc = by_day[day];
            acc.wolf_f.push_back(wf_fault.wolf_lye);
            if (std::isfinite(wolf_ctrl))
              acc.wolf_c.push_back(wolf_ctrl);
            if (std::isfinite(wf_fault.ros_short))
              acc.ros_f.push_back(wf_fault.ros_short);
            if (std::isfinite(wf_fault.samp_ent))
              acc.se_f.push_back(wf_fault.samp_ent);
            if (wf_fault.corr_dim > 0.0)
              acc.cd_f.push_back(wf_fault.corr_dim);
          }
        }
      }

      if (by_day.empty())
        return;

      std::vector<double> all_wolf_pre;
      std::vector<double> all_cb_pre;
      std::vector<double> pre_ros;
      std::vector<double> post_wolf;
      std::vector<double> post_ros;

      for (auto &[day, acc] : by_day) {
        if (acc.wolf_f.empty())
          continue;
        DailyStationLyapunov d;
        d.day_mjd = day;
        d.n_windows = acc.wolf_f.size();
        d.wolf_fault = linalg::median(acc.wolf_f);
        d.wolf_ctrl =
            acc.wolf_c.empty() ? d.wolf_fault : linalg::median(acc.wolf_c);
        d.ros_short_fault = acc.ros_f.empty() ? 0.0 : linalg::median(acc.ros_f);
        d.samp_ent_fault = acc.se_f.empty() ? 0.0 : linalg::median(acc.se_f);
        d.corr_dim_fault = acc.cd_f.empty() ? 0.0 : linalg::median(acc.cd_f);
        sum.daily.push_back(d);

        if (day < ev_day) {
          all_wolf_pre.push_back(d.wolf_fault);
          all_cb_pre.push_back(d.wolf_fault - d.wolf_ctrl);
          pre_ros.push_back(d.ros_short_fault);
        } else {
          post_wolf.push_back(d.wolf_fault);
          post_ros.push_back(d.ros_short_fault);
        }
      }

      std::vector<double> series_w;
      std::vector<double> series_cb;
      for (const auto &d : sum.daily) {
        series_w.push_back(d.wolf_fault);
        series_cb.push_back(d.wolf_fault - d.wolf_ctrl);
      }
      const auto z_w = detail::compute_robust_z(series_w, all_wolf_pre);
      const auto z_cb = detail::compute_robust_z(series_cb, all_cb_pre);

      std::vector<double> pre_scores;
      std::vector<std::uint8_t> pre_labels;

      for (std::size_t i = 0; i < sum.daily.size(); ++i) {
        sum.daily[i].z_iqr_wolf = z_w[i];
        sum.daily[i].z_iqr_crossband = z_cb[i];
        const int day = sum.daily[i].day_mjd;
        if (day < ev_day) {
          const bool in_horizon = (day >= ev_day - 3);
          if (in_horizon) {
            sum.pre_horizon_max_z =
                std::max(sum.pre_horizon_max_z, std::abs(z_w[i]));
            sum.pre_horizon_crossband_z =
                std::max(sum.pre_horizon_crossband_z, std::abs(z_cb[i]));
          }
          pre_scores.push_back(std::abs(z_w[i]) + 0.5 * std::abs(z_cb[i]));
          pre_labels.push_back(in_horizon ? 1 : 0);
        }
      }

      if (!all_wolf_pre.empty())
        sum.pre_wolf_median = linalg::median(all_wolf_pre);
      if (!post_wolf.empty())
        sum.post_wolf_median = linalg::median(post_wolf);
      sum.delta_wolf_post_minus_pre =
          sum.post_wolf_median - sum.pre_wolf_median;

      if (!pre_ros.empty())
        sum.pre_ros_short_median = linalg::median(pre_ros);
      if (!post_ros.empty())
        sum.post_ros_short_median = linalg::median(post_ros);
      sum.delta_ros_short =
          sum.post_ros_short_median - sum.pre_ros_short_median;

      sum.station_auc = detail::roc_auc(pre_scores, pre_labels);
      summaries[s_idx] = std::move(sum);
    });
  }

  for (auto &th : workers)
    th.join();

  for (auto &s : summaries) {
    if (!s.daily.empty())
      out.stations.push_back(std::move(s));
  }
  if (out.stations.empty())
    return out;

  out.has_seismic = true;

  // Daily multi-station array dispersion (`array_disp`)
  std::map<int, std::vector<double>> day_zs;
  for (const auto &s : out.stations) {
    out.mean_abs_delta_wolf += std::abs(s.delta_wolf_post_minus_pre);
    for (const auto &d : s.daily) {
      day_zs[d.day_mjd].push_back(d.z_iqr_wolf);
    }
  }
  out.mean_abs_delta_wolf /= static_cast<double>(out.stations.size());

  std::vector<double> arr_pre_scores;
  std::vector<std::uint8_t> arr_pre_labels;

  for (const auto &[day, zs] : day_zs) {
    DailyArrayDispersion ad;
    ad.day_mjd = day;
    ad.n_active_stations = zs.size();
    double m = 0.0;
    double mx = 0.0;
    for (double v : zs) {
      m += v;
      mx = std::max(mx, std::abs(v));
    }
    m /= static_cast<double>(zs.size());
    double s2 = 0.0;
    for (double v : zs) {
      const double dv = v - m;
      s2 += dv * dv;
    }
    ad.mean_z = m;
    ad.spatial_std_z = (zs.size() > 1)
                           ? std::sqrt(s2 / static_cast<double>(zs.size() - 1))
                           : std::abs(m);
    ad.max_z = mx;
    out.array_daily.push_back(ad);

    if (day < ev_day) {
      arr_pre_scores.push_back(ad.spatial_std_z + 0.5 * ad.max_z);
      arr_pre_labels.push_back((day >= ev_day - 3) ? 1 : 0);
    }
  }

  out.array_pre_auc = detail::roc_auc(arr_pre_scores, arr_pre_labels);

  // Circular time-slide null p-value on `arr_pre_scores`
  if (arr_pre_scores.size() >= 6) {
    std::size_t ge_count = 0;
    const std::size_t n = arr_pre_scores.size();
    std::vector<std::uint8_t> shifted(n);
    for (std::size_t shift = 0; shift < n; ++shift) {
      for (std::size_t i = 0; i < n; ++i)
        shifted[(i + shift) % n] = arr_pre_labels[i];
      if (detail::roc_auc(arr_pre_scores, shifted) >= out.array_pre_auc - 1e-12)
        ++ge_count;
    }
    out.array_null_p_value =
        static_cast<double>(ge_count) / static_cast<double>(n);
  }

  // Spatial interpolation of station L-seis metrics onto Grid2D
  constexpr double kSigmaKm = 55.0;
  constexpr double kInv2Sig2 = 1.0 / (2.0 * kSigmaKm * kSigmaKm);
  std::vector<linalg::Vec2> st_enu(out.stations.size());
  for (std::size_t i = 0; i < out.stations.size(); ++i) {
    st_enu[i] =
        grid.proj.to_enu(out.stations[i].lat_deg, out.stations[i].lon_deg);
  }

  double max_pre = 0.0, max_post = 0.0, max_chg = 0.0;
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const std::size_t idx = r * grid.n_lon + c;
      const auto gpt = grid.enu_at(r, c);
      double w_sum = 0.0;
      double v_pre = 0.0, v_post = 0.0, v_chg = 0.0;
      for (std::size_t i = 0; i < out.stations.size(); ++i) {
        const double dx = gpt.x - st_enu[i].x;
        const double dy = gpt.y - st_enu[i].y;
        const double w = std::exp(-(dx * dx + dy * dy) * kInv2Sig2);
        w_sum += w;
        const double s_pre = out.stations[i].pre_horizon_max_z +
                             0.5 * out.stations[i].pre_horizon_crossband_z;
        const double s_chg =
            std::abs(out.stations[i].delta_wolf_post_minus_pre) +
            10.0 * std::abs(out.stations[i].delta_ros_short);
        const double s_post = s_pre + s_chg;
        v_pre += w * s_pre;
        v_post += w * s_post;
        v_chg += w * s_chg;
      }
      if (w_sum > 1e-12) {
        // Modulate by proximity to station network so far-field edges decay
        const double env = std::min(1.0, w_sum);
        out.grid_l_seis_pre[idx] = (v_pre / w_sum) * env;
        out.grid_l_seis_post[idx] = (v_post / w_sum) * env;
        out.grid_l_seis_change[idx] = (v_chg / w_sum) * env;
        max_pre = std::max(max_pre, out.grid_l_seis_pre[idx]);
        max_post = std::max(max_post, out.grid_l_seis_post[idx]);
        max_chg = std::max(max_chg, out.grid_l_seis_change[idx]);
      }
    }
  }

  auto norm_grid = [](std::vector<double> &g, double mx) {
    if (mx > 1e-12) {
      for (double &v : g)
        v = std::clamp(v / mx, 0.0, 1.0);
    }
  };
  norm_grid(out.grid_l_seis_pre, max_pre);
  norm_grid(out.grid_l_seis_post, max_post);
  norm_grid(out.grid_l_seis_change, max_chg);

  // Extract principal ridge axis of `grid_l_seis_change` (top 20% cells)
  double sw = 0.0, mx_x = 0.0, mx_y = 0.0;
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const double val = out.grid_l_seis_change[r * grid.n_lon + c];
      if (val < 0.75)
        continue;
      const auto pt = grid.enu_at(r, c);
      sw += val;
      mx_x += val * pt.x;
      mx_y += val * pt.y;
    }
  }
  if (sw > 0.0) {
    mx_x /= sw;
    mx_y /= sw;
    double cxx = 0.0, cxy = 0.0, cyy = 0.0;
    for (std::size_t r = 0; r < grid.n_lat; ++r) {
      for (std::size_t c = 0; c < grid.n_lon; ++c) {
        const double val = out.grid_l_seis_change[r * grid.n_lon + c];
        if (val < 0.75)
          continue;
        const auto pt = grid.enu_at(r, c);
        const double dx = pt.x - mx_x;
        const double dy = pt.y - mx_y;
        cxx += val * dx * dx;
        cxy += val * dx * dy;
        cyy += val * dy * dy;
      }
    }
    const auto eig = linalg::sym_eig2(cxx / sw, cxy / sw, cyy / sw);
    const auto maj = eig.vectors[1];
    double s_deg = geo::rad2deg(std::atan2(maj.x, maj.y));
    if (s_deg < 0.0)
      s_deg += 360.0;
    out.ridge_strike_deg = s_deg;
    const double half_len =
        std::clamp(2.0 * std::sqrt(std::max(1.0, eig.values[1])), 15.0, 65.0);
    const auto p0 =
        grid.proj.to_latlon(mx_x - half_len * maj.x, mx_y - half_len * maj.y);
    const auto p1 =
        grid.proj.to_latlon(mx_x + half_len * maj.x, mx_y + half_len * maj.y);
    out.ridge_start_lat = p0.x;
    out.ridge_start_lon = p0.y;
    out.ridge_end_lat = p1.x;
    out.ridge_end_lon = p1.y;
  }

  return out;
}

} // namespace chaosmap::anomaly
