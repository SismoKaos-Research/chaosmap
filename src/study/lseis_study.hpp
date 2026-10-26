#pragma once

// Surrogate-controlled test of the broadband Lyapunov (`L-seis`) change across
// a mainshock. For every background window it records the Wolf exponent, the
// Wolf exponents of IAAFT surrogates of that window (what its spectrum and
// amplitude distribution alone produce), and simple spectral covariates (mean
// period, fault- and control-band RMS). Per station it then splits the
// post-minus-pre change of the Wolf exponent into the part its surrogates
// predict and the excess, and refits the change against the covariates, all
// with day-block bootstrap confidence intervals.

#include "anomaly/array_anomaly.hpp"
#include "chaos/lyapunov.hpp"
#include "chaos/surrogates.hpp"
#include "core/config.hpp"
#include "core/linalg.hpp"
#include "dsp/filter.hpp"
#include "geo/projection.hpp"
#include "io/catalog.hpp"
#include "io/zip_mseed.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace chaosmap::study {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Defaults mirror the window selection and features of
// `anomaly::analyze_broadband_array`, so the study tests the same numbers the
// fusion pipeline uses.
struct LseisStudyParams {
  double pre_days{8.0};
  double post_days{7.0};
  double win_sec{600.0};
  std::size_t stride_windows{18}; // one window every 3 h
  double target_fs{5.0};
  double fault_fmin{0.05};
  double fault_fmax{0.50};
  double ctrl_fmin{0.5};
  double ctrl_fmax{1.8};
  double exclude_before_days{0.05}; // windows near the origin are skipped
  double exclude_after_days{0.35};
  double mask_min_mag{4.3};
  int tau{5};
  int dim{5};
  int evolve{5};
  // 39 surrogates make "outside the surrogate range" a two-sided 5% test.
  int n_surrogates{39};
  int n_bootstrap{2000};
  std::uint64_t seed{1};
};

struct WindowRecord {
  std::size_t station_idx{};
  double start_mjd{};
  int rel_day{}; // floor(start - origin) in days; < 0 before the mainshock
  bool post{false};
  double wolf{kNaN};
  double surr_mean{kNaN};
  double surr_sd{kNaN};
  double surr_z{kNaN};
  double surr_rank{kNaN};
  int n_surr{0};
  double mean_period_s{kNaN};
  double log10_rms_fault{kNaN};
  double log10_rms_ctrl{kNaN};
};

// Point estimate with a 95% day-block bootstrap interval.
struct Effect {
  double estimate{kNaN};
  double ci_lo{kNaN};
  double ci_hi{kNaN};
};

struct StationStudy {
  std::string station;
  std::string channel;
  double lat_deg{};
  double lon_deg{};
  double dist_km{};
  std::size_t n_pre{0};
  std::size_t n_post{0};
  // Windows whose Wolf exponent lies outside its surrogates' range
  double frac_nonlinear_pre{kNaN};
  double frac_nonlinear_post{kNaN};
  double r2_surr{kNaN};       // R^2 of Wolf on its surrogate mean
  double r2_covariates{kNaN}; // R^2 of Wolf on mean period and band RMS
  Effect delta_wolf;          // mean(post) - mean(pre)
  Effect delta_surr_mean;     // part of `delta_wolf` the surrogates predict
  Effect delta_excess;        // change of (Wolf - surrogate mean)
  Effect delta_wolf_adj;      // post coefficient of Wolf ~ covariates + post
};

struct LseisStudyResult {
  LseisStudyParams params;
  double origin_mjd{};
  std::vector<StationStudy> stations;
  std::vector<WindowRecord> windows; // sorted by station, then time
};

namespace detail {

[[nodiscard]] inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

[[nodiscard]] inline double sample_std(std::span<const double> x) noexcept {
  if (x.size() < 2)
    return 0.0;
  double mean = 0.0;
  for (double v : x)
    mean += v;
  mean /= static_cast<double>(x.size());
  double s2 = 0.0;
  for (double v : x)
    s2 += (v - mean) * (v - mean);
  return std::sqrt(s2 / static_cast<double>(x.size() - 1));
}

struct OlsFit {
  std::vector<double> beta;
  double r2{kNaN};
};

// Ordinary least squares of `y` on the columns of `X` (which should include
// an intercept column).
[[nodiscard]] inline OlsFit ols(const linalg::Mat &X,
                                std::span<const double> y) {
  OlsFit out;
  if (X.rows() <= X.cols() || y.size() != X.rows())
    return out;
  const auto fit = linalg::solve_wls(X, y, {});
  if (fit.rank < static_cast<int>(X.cols()))
    return out;
  out.beta = fit.x;
  double mean = 0.0;
  for (double v : y)
    mean += v;
  mean /= static_cast<double>(y.size());
  double tss = 0.0;
  for (double v : y)
    tss += (v - mean) * (v - mean);
  out.r2 = (tss > 0.0) ? 1.0 - fit.rss / tss : kNaN;
  return out;
}

// {delta_wolf, delta_surr_mean, delta_excess, delta_wolf_adj} for one set of
// windows (possibly with repeats, as drawn by the bootstrap).
[[nodiscard]] inline std::array<double, 4>
station_effects(std::span<const WindowRecord *const> w) {
  std::array<double, 4> out{kNaN, kNaN, kNaN, kNaN};
  double sum_wolf[2]{}, sum_surr[2]{};
  std::size_t n[2]{};
  for (const auto *r : w) {
    const int g = r->post ? 1 : 0;
    sum_wolf[g] += r->wolf;
    sum_surr[g] += r->surr_mean;
    ++n[g];
  }
  if (n[0] == 0 || n[1] == 0)
    return out;
  const auto dn0 = static_cast<double>(n[0]);
  const auto dn1 = static_cast<double>(n[1]);
  out[0] = sum_wolf[1] / dn1 - sum_wolf[0] / dn0;
  out[1] = sum_surr[1] / dn1 - sum_surr[0] / dn0;
  out[2] = out[0] - out[1];

  linalg::Mat X(w.size(), 5);
  std::vector<double> y(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    X(i, 0) = 1.0;
    X(i, 1) = w[i]->mean_period_s;
    X(i, 2) = w[i]->log10_rms_fault;
    X(i, 3) = w[i]->log10_rms_ctrl;
    X(i, 4) = w[i]->post ? 1.0 : 0.0;
    y[i] = w[i]->wolf;
  }
  const auto fit = ols(X, y);
  if (!fit.beta.empty())
    out[3] = fit.beta[4];
  return out;
}

// Fills the statistics of `st` from its windows `w`.
inline void summarize_station(StationStudy &st,
                              std::span<const WindowRecord *const> w,
                              const LseisStudyParams &p,
                              std::size_t station_idx) {
  std::size_t nl[2]{};
  for (const auto *r : w) {
    const int g = r->post ? 1 : 0;
    (g ? st.n_post : st.n_pre) += 1;
    if (r->n_surr >= 2 && (r->surr_rank <= 0.0 || r->surr_rank >= 1.0))
      ++nl[g];
  }
  if (st.n_pre > 0)
    st.frac_nonlinear_pre =
        static_cast<double>(nl[0]) / static_cast<double>(st.n_pre);
  if (st.n_post > 0)
    st.frac_nonlinear_post =
        static_cast<double>(nl[1]) / static_cast<double>(st.n_post);

  // How much of the Wolf exponent's variance the spectrum explains
  {
    linalg::Mat Xs(w.size(), 2);
    linalg::Mat Xc(w.size(), 4);
    std::vector<double> y(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) {
      Xs(i, 0) = 1.0;
      Xs(i, 1) = w[i]->surr_mean;
      Xc(i, 0) = 1.0;
      Xc(i, 1) = w[i]->mean_period_s;
      Xc(i, 2) = w[i]->log10_rms_fault;
      Xc(i, 3) = w[i]->log10_rms_ctrl;
      y[i] = w[i]->wolf;
    }
    st.r2_surr = ols(Xs, y).r2;
    st.r2_covariates = ols(Xc, y).r2;
  }

  const auto est = station_effects(w);

  // Day-block bootstrap: resample whole days (windows within a day are
  // correlated), separately before and after the mainshock.
  std::map<int, std::vector<const WindowRecord *>> days[2];
  for (const auto *r : w)
    days[r->post ? 1 : 0][r->rel_day].push_back(r);
  std::vector<const std::vector<const WindowRecord *> *> day_list[2];
  for (int g = 0; g < 2; ++g)
    for (const auto &[d, v] : days[g])
      day_list[g].push_back(&v);

  std::array<std::vector<double>, 4> reps;
  if (!day_list[0].empty() && !day_list[1].empty()) {
    std::mt19937_64 rng(splitmix64(p.seed ^ (0xB0075ULL + station_idx)));
    std::vector<const WindowRecord *> sample;
    for (int b = 0; b < p.n_bootstrap; ++b) {
      sample.clear();
      for (int g = 0; g < 2; ++g) {
        std::uniform_int_distribution<std::size_t> pick(
            0, day_list[g].size() - 1);
        for (std::size_t k = 0; k < day_list[g].size(); ++k) {
          const auto &day = *day_list[g][pick(rng)];
          sample.insert(sample.end(), day.begin(), day.end());
        }
      }
      const auto e = station_effects(sample);
      for (std::size_t k = 0; k < 4; ++k)
        if (std::isfinite(e[k]))
          reps[k].push_back(e[k]);
    }
  }

  Effect *effects[4] = {&st.delta_wolf, &st.delta_surr_mean, &st.delta_excess,
                        &st.delta_wolf_adj};
  for (std::size_t k = 0; k < 4; ++k) {
    effects[k]->estimate = est[k];
    if (reps[k].size() >= 20) {
      effects[k]->ci_lo = linalg::quantile(reps[k], 0.025);
      effects[k]->ci_hi = linalg::quantile(reps[k], 0.975);
    }
  }
}

} // namespace detail

[[nodiscard]] inline LseisStudyResult
run_lseis_study(const core::CaseConfig &cfg,
                std::span<const io::CatalogEvent> events,
                const LseisStudyParams &p = {}) {
  LseisStudyResult out;
  out.params = p;
  out.origin_mjd =
      (cfg.event.origin_epoch_s != 0)
          ? 40587.0 + static_cast<double>(cfg.event.origin_epoch_s) / 86400.0
          : cfg.event.mjd;
  const double origin_mjd = out.origin_mjd;
  const double min_mjd = origin_mjd - p.pre_days;
  const double max_mjd = origin_mjd + p.post_days;
  const double win_days = p.win_sec / 86400.0;

  std::error_code ec;
  if (cfg.paths.broadband_dir.empty() ||
      !std::filesystem::exists(cfg.paths.broadband_dir, ec))
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

  // 1. Load each station's selected channel (one thread per station)
  struct StationData {
    StationStudy info;
    std::vector<io::mseed::ContinuousTrace> traces;
    bool ok{false};
  };
  std::vector<StationData> data(sta_dirs.size());
  {
    std::vector<std::thread> loaders;
    for (std::size_t s = 0; s < sta_dirs.size(); ++s) {
      loaders.emplace_back([&, s]() {
        auto &sd = data[s];
        sd.info.station = sta_dirs[s].filename().string();
        for (const auto &sc : coords) {
          if (sc.station == sd.info.station) {
            sd.info.lat_deg = sc.lat_deg;
            sd.info.lon_deg = sc.lon_deg;
            break;
          }
        }
        if (sd.info.lat_deg == 0.0 && sd.info.lon_deg == 0.0)
          return;
        sd.info.dist_km = geo::haversine_km(sd.info.lat_deg, sd.info.lon_deg,
                                            cfg.event.lat, cfg.event.lon);
        sd.traces = io::load_station_waveforms(sta_dirs[s], min_mjd, max_mjd);
        if (sd.traces.empty())
          return;
        sd.info.channel = sd.traces.front().channel;
        sd.ok = true;
      });
    }
    for (auto &th : loaders)
      th.join();
  }
  std::erase_if(data, [](const StationData &sd) { return !sd.ok; });

  // 2. Background windows, selected as in `analyze_broadband_array`
  struct Job {
    std::size_t station_idx;
    const io::mseed::ContinuousTrace *trace;
    std::size_t offset;
    std::size_t win_n;
    double start_mjd;
  };
  std::vector<Job> jobs;
  for (std::size_t s = 0; s < data.size(); ++s) {
    for (const auto &tr : data[s].traces) {
      if (tr.sample_rate_hz < 10.0)
        continue;
      const auto win_n =
          static_cast<std::size_t>(std::llround(p.win_sec * tr.sample_rate_hz));
      const std::size_t stride_n = win_n * p.stride_windows;
      for (std::size_t off = 0; off + win_n <= tr.data.size();
           off += stride_n) {
        const double w_mjd =
            tr.start_mjd +
            (static_cast<double>(off) / tr.sample_rate_hz) / 86400.0;
        if (w_mjd < min_mjd || w_mjd > max_mjd)
          continue;
        if (w_mjd + win_days >= origin_mjd - p.exclude_before_days &&
            w_mjd <= origin_mjd + p.exclude_after_days)
          continue;
        if (anomaly::detail::window_overlaps_catalog(
                w_mjd, w_mjd + win_days, events, p.mask_min_mag))
          continue;
        jobs.push_back({s, &tr, off, win_n, w_mjd});
      }
    }
  }

  // 3. Features and surrogate tests, in parallel over windows. Each window's
  //    surrogate seed depends only on its station and start time, so results
  //    do not depend on scheduling.
  std::vector<WindowRecord> recs(jobs.size());
  std::vector<std::uint8_t> valid(jobs.size(), 0);
  {
    std::atomic<std::size_t> next{0};
    const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < n_threads; ++t) {
      workers.emplace_back([&]() {
        for (std::size_t j = next++; j < jobs.size(); j = next++) {
          const auto &job = jobs[j];
          const std::span<const double> raw(job.trace->data.data() + job.offset,
                                            job.win_n);
          const double fs = job.trace->sample_rate_hz;
          const auto x = dsp::preprocess_window(raw, fs, p.target_fs,
                                                p.fault_fmin, p.fault_fmax);
          if (x.size() < 200)
            continue;
          const double rms_fault = detail::sample_std(x);
          const auto x_ctrl = dsp::preprocess_window(raw, fs, p.target_fs,
                                                     p.ctrl_fmin, p.ctrl_fmax);
          const double rms_ctrl = detail::sample_std(x_ctrl);
          if (rms_fault <= 1e-20 || rms_ctrl <= 1e-20)
            continue;

          auto wolf = [&](std::span<const double> s) {
            return chaos::wolf_lye_core(chaos::zscore_normalize(s), p.target_fs,
                                        p.tau, p.dim, p.evolve);
          };
          const std::uint64_t seed = detail::splitmix64(
              p.seed ^ detail::splitmix64(
                           (static_cast<std::uint64_t>(job.station_idx) << 40) ^
                           static_cast<std::uint64_t>(
                               std::llround(job.start_mjd * 86400.0))));
          const auto st = chaos::surrogate_test(x, p.n_surrogates, seed, wolf);
          if (!std::isfinite(st.value))
            continue;

          auto &r = recs[j];
          r.station_idx = job.station_idx;
          r.start_mjd = job.start_mjd;
          r.rel_day = static_cast<int>(std::floor(job.start_mjd - origin_mjd));
          r.post = job.start_mjd >= origin_mjd;
          r.wolf = st.value;
          r.surr_mean = st.surr_mean;
          r.surr_sd = st.surr_sd;
          r.surr_z = st.z;
          r.surr_rank = st.rank_frac;
          r.n_surr = st.n_surr;
          r.mean_period_s = dsp::spectral_mean_period(x, p.target_fs);
          r.log10_rms_fault = std::log10(rms_fault);
          r.log10_rms_ctrl = std::log10(rms_ctrl);
          valid[j] = std::isfinite(r.surr_mean) ? 1 : 0;
        }
      });
    }
    for (auto &th : workers)
      th.join();
  }
  for (std::size_t j = 0; j < recs.size(); ++j)
    if (valid[j])
      out.windows.push_back(recs[j]);
  std::ranges::sort(out.windows, [](const WindowRecord &a,
                                    const WindowRecord &b) {
    return (a.station_idx != b.station_idx) ? a.station_idx < b.station_idx
                                            : a.start_mjd < b.start_mjd;
  });

  // 4. Per-station statistics
  for (std::size_t s = 0; s < data.size(); ++s) {
    std::vector<const WindowRecord *> w;
    for (const auto &r : out.windows)
      if (r.station_idx == s)
        w.push_back(&r);
    auto st = std::move(data[s].info);
    detail::summarize_station(st, w, p, s);
    out.stations.push_back(std::move(st));
  }
  return out;
}

} // namespace chaosmap::study
