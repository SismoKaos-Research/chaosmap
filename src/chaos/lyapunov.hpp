#pragma once

// Exact C++23 port of `chaos/src/chaos/chaos_algorithms.py` and
// `chaotic_features.py` (Wolf 1985 largest Lyapunov exponent, Rosenstein 1993
// short/long divergence slope + R^2, Sample Entropy, Correlation Dimension).

#include "dsp/filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

namespace chaosmap::chaos {

struct DivergenceFit {
  double slope{std::numeric_limits<double>::quiet_NaN()};
  double r2{std::numeric_limits<double>::quiet_NaN()};
  int n_points{0};
};

struct RosensteinResult {
  DivergenceFit short_fit;
  DivergenceFit long_fit;
  std::vector<double> ave_ln_div;
};

struct WindowFeatures {
  double start_mjd{};
  double mean_period_s{};
  double raw_std{};
  double wolf_lye{std::numeric_limits<double>::quiet_NaN()};
  double ros_short{std::numeric_limits<double>::quiet_NaN()};
  double ros_short_r2{std::numeric_limits<double>::quiet_NaN()};
  double ros_long{std::numeric_limits<double>::quiet_NaN()};
  double ros_long_r2{std::numeric_limits<double>::quiet_NaN()};
  double samp_ent{std::numeric_limits<double>::quiet_NaN()};
  double corr_dim{std::numeric_limits<double>::quiet_NaN()};
};

// Z-score normalizes `x` using sample standard deviation (`ddof = 1`).
[[nodiscard]] inline std::vector<double>
zscore_normalize(std::span<const double> x) {
  std::vector<double> out(x.begin(), x.end());
  if (out.size() < 2)
    return out;
  double mean = 0.0;
  for (double v : out)
    mean += v;
  mean /= static_cast<double>(out.size());
  double s2 = 0.0;
  for (double v : out) {
    const double d = v - mean;
    s2 += d * d;
  }
  const double sd = std::sqrt(s2 / static_cast<double>(out.size() - 1));
  if (sd <= 1e-30)
    return out;
  for (double &v : out)
    v = (v - mean) / sd;
  return out;
}

// Wolf (1985) maximum Lyapunov exponent matching
// `chaos_algorithms.py:wolf_lye_core`.
[[nodiscard]] inline double wolf_lye_core(std::span<const double> x, double fs,
                                          int tau, int dim, int evolve) {
  const int N = static_cast<int>(x.size());
  const int M = N - (dim - 1) * tau;
  const int NPT = M - evolve;
  if (NPT < 1 || dim < 1 || tau < 1 || evolve < 1 || fs <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();

  auto [mn_it, mx_it] = std::ranges::minmax_element(x);
  const double SCALEMX = (*mx_it - *mn_it) / 10.0;
  double ANGLMX = 30.0 * std::numbers::pi / 180.0;
  const double DT = 1.0 / fs;

  const int Y_rows = NPT + evolve;
  auto y_at = [&](int r, int c) noexcept -> double {
    return x[static_cast<std::size_t>(r + c * tau)];
  };
  auto dist_rows = [&](int r1, int r2) noexcept -> double {
    double s = 0.0;
    for (int c = 0; c < dim; ++c) {
      const double d = y_at(r1, c) - y_at(r2, c);
      s += d * d;
    }
    return std::sqrt(s);
  };

  int current_point_pair = -1;
  double best_d = std::numeric_limits<double>::infinity();
  for (int j = 11; j < NPT; ++j) {
    const double d = dist_rows(0, j);
    if (d > 0.0 && d < best_d) {
      best_d = d;
      current_point_pair = j;
    }
  }
  if (current_point_pair < 0 || !std::isfinite(best_d))
    return std::numeric_limits<double>::quiet_NaN();

  auto wolf_next_point = [&](int flag, int current_point, int pair_idx,
                             double &anglmx) -> int {
    const int ep = current_point + evolve;
    const int end_idx = (pair_idx + evolve < Y_rows) ? (pair_idx + evolve)
                                                     : (pair_idx + evolve - 1);
    const double end_dist = dist_rows(ep, end_idx);
    const double cos_thresh = std::cos(anglmx);

    int cand = -1;
    double best_pot = std::numeric_limits<double>::infinity();
    int fallback = -1;
    double best_tmp = std::numeric_limits<double>::infinity();

    for (int j = 0; j < NPT; ++j) {
      if (j >= std::max(0, ep - 10) && j < std::min(NPT, ep + 11))
        continue;
      double d2 = 0.0;
      double dot = 0.0;
      for (int c = 0; c < dim; ++c) {
        const double diff_c = y_at(ep, c) - y_at(j, c);
        const double vcurr_c = y_at(ep, c) - y_at(end_idx, c);
        d2 += diff_c * diff_c;
        dot += diff_c * vcurr_c;
      }
      const double yd = std::sqrt(d2);
      if (yd <= 0.0)
        continue;
      if (yd < best_tmp) {
        best_tmp = yd;
        fallback = j;
      }
      if (flag == 0 && end_dist > 0.0) {
        const double cos_t = std::abs(dot / (yd * end_dist));
        if (cos_t > cos_thresh && yd < best_pot) {
          best_pot = yd;
          cand = j;
        }
      }
    }

    if (flag == 0 && cand >= 0 && best_pot <= SCALEMX) {
      anglmx = 30.0 * std::numbers::pi / 180.0;
      return cand;
    }
    if (fallback >= 0 && std::isfinite(best_tmp))
      return fallback;
    return pair_idx;
  };

  int ITS = 0;
  double distSUM = 0.0;
  double LyE = 0.0;

  for (int i = 0; i < NPT; i += evolve) {
    const int ep = i + evolve;
    const bool safe = (current_point_pair + evolve < Y_rows) && (ep < Y_rows);
    const int pair_ep = safe ? (current_point_pair + evolve)
                             : (current_point_pair + evolve - 1);

    const double start_dist = dist_rows(i, current_point_pair);
    const double end_dist = dist_rows(ep, pair_ep);

    if (start_dist > 0.0 && end_dist > 0.0) {
      distSUM +=
          std::log2(end_dist / start_dist) / (static_cast<double>(evolve) * DT);
      ++ITS;
      LyE = distSUM / static_cast<double>(ITS);
    }

    if (end_dist < SCALEMX) {
      current_point_pair += evolve;
      if (current_point_pair > NPT) {
        current_point_pair -= evolve;
        current_point_pair = wolf_next_point(1, i, current_point_pair, ANGLMX);
      }
    } else {
      current_point_pair = wolf_next_point(0, i, current_point_pair, ANGLMX);
    }
  }
  return (ITS > 0) ? LyE : std::numeric_limits<double>::quiet_NaN();
}

// Least-squares slope of the mean log-divergence curve over `[start, end]`
// mean periods, matching `chaos_algorithms.py:fit_log_divergence`.
[[nodiscard]] inline DivergenceFit
fit_log_divergence(std::span<const double> ave_ln_div, double fs,
                   double mean_period, double start_periods,
                   double end_periods) noexcept {
  DivergenceFit out;
  int usable = 0;
  for (int i = static_cast<int>(ave_ln_div.size()) - 1; i >= 0; --i) {
    if (ave_ln_div[static_cast<std::size_t>(i)] != 0.0) {
      usable = i + 1;
      break;
    }
  }
  const int lo =
      static_cast<int>(std::llround(start_periods * mean_period * fs));
  const int hi = static_cast<int>(std::llround(end_periods * mean_period * fs));
  if (lo < 0 || hi <= lo || hi >= usable || fs <= 0.0 || mean_period <= 0.0)
    return out;

  const int n = hi - lo + 1;
  out.n_points = n;
  if (n < 2)
    return out;

  const double inv_scale = 1.0 / (fs * mean_period);
  double mean_t = 0.0;
  double mean_y = 0.0;
  for (int k = lo; k <= hi; ++k) {
    mean_t += static_cast<double>(k) * inv_scale;
    mean_y += ave_ln_div[static_cast<std::size_t>(k)];
  }
  mean_t /= static_cast<double>(n);
  mean_y /= static_cast<double>(n);

  double s_tt = 0.0;
  double s_ty = 0.0;
  double s_yy = 0.0;
  for (int k = lo; k <= hi; ++k) {
    const double dt = static_cast<double>(k) * inv_scale - mean_t;
    const double dy = ave_ln_div[static_cast<std::size_t>(k)] - mean_y;
    s_tt += dt * dt;
    s_ty += dt * dy;
    s_yy += dy * dy;
  }
  if (s_tt <= 0.0)
    return out;

  out.slope = s_ty / s_tt;
  const double ss_res = std::max(0.0, s_yy - out.slope * s_ty);
  out.r2 = (s_yy > 0.0) ? (1.0 - ss_res / s_yy)
                        : std::numeric_limits<double>::quiet_NaN();
  return out;
}

// Rosenstein (1993) divergence curve + short/long slope fits matching
// `chaos_algorithms.py:rosenstein_lye_core`. Uses 1D sorted projection pruning
// along the first embedding coordinate so exact Euclidean nearest-neighbour
// search outside Theiler band `|i - j| > (dim - 1)*tau` runs in sub-millisecond
// time.
[[nodiscard]] inline RosensteinResult
rosenstein_lye_core(std::span<const double> x, double fs, int tau, int dim,
                    double mean_period, double short_start = 0.0,
                    double short_end = 1.0, double long_start = 1.0,
                    double long_end = 4.0) {
  RosensteinResult res;
  const int N = static_cast<int>(x.size());
  const int M = N - (dim - 1) * tau;
  if (M < 2 || dim < 1 || tau < 1)
    return res;

  const int band = (dim - 1) * tau;

  // Sort trajectory indices by first coordinate x[i] for fast branch-and-bound
  // exact nearest-neighbour search
  std::vector<int> order(static_cast<std::size_t>(M));
  for (int i = 0; i < M; ++i)
    order[static_cast<std::size_t>(i)] = i;
  std::ranges::sort(order, [&](int a, int b) {
    return x[static_cast<std::size_t>(a)] < x[static_cast<std::size_t>(b)];
  });
  std::vector<int> rank(static_cast<std::size_t>(M));
  for (int pos = 0; pos < M; ++pos)
    rank[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])] = pos;

  auto dist2 = [&](int i, int j) noexcept -> double {
    double s = 0.0;
    for (int c = 0; c < dim; ++c) {
      const double d = x[static_cast<std::size_t>(i + c * tau)] -
                       x[static_cast<std::size_t>(j + c * tau)];
      s += d * d;
    }
    return s;
  };

  std::vector<int> neighbours(static_cast<std::size_t>(M), 0);
  for (int i = 0; i < M; ++i) {
    const int r0 = rank[static_cast<std::size_t>(i)];
    const double xi = x[static_cast<std::size_t>(i)];
    double best_d2 = std::numeric_limits<double>::infinity();
    int best_j = 0;

    int left = r0 - 1;
    int right = r0 + 1;
    while (left >= 0 || right < M) {
      bool go_left = false;
      if (left >= 0 && right < M) {
        const double dl =
            xi -
            x[static_cast<std::size_t>(order[static_cast<std::size_t>(left)])];
        const double dr = x[static_cast<std::size_t>(
                              order[static_cast<std::size_t>(right)])] -
                          xi;
        go_left = (dl <= dr);
      } else {
        go_left = (left >= 0);
      }

      const int pos = go_left ? left-- : right++;
      const int j = order[static_cast<std::size_t>(pos)];
      const double dx0 = xi - x[static_cast<std::size_t>(j)];
      if (dx0 * dx0 > best_d2) {
        // Both sides sorted by x[j]; if current side exceeds best_d2, drain
        // only the other side if needed
        if (go_left) {
          left = -1;
        } else {
          right = M;
        }
        continue;
      }
      if (std::abs(i - j) <= band)
        continue;
      const double d2 = dist2(i, j);
      if (d2 < best_d2 || (d2 == best_d2 && j < best_j)) {
        best_d2 = d2;
        best_j = j;
      }
    }
    neighbours[static_cast<std::size_t>(i)] = best_j;
  }

  // Accumulate mean log divergence across steps k = 0 .. M-1
  res.ave_ln_div.assign(static_cast<std::size_t>(M), 0.0);
  std::vector<std::int64_t> count(static_cast<std::size_t>(M), 0);

  for (int i = 0; i < M; ++i) {
    const int nn = neighbours[static_cast<std::size_t>(i)];
    const int end = std::min(M - i, M - nn);
    for (int k = 0; k < end; ++k) {
      const double d = std::sqrt(dist2(i + k, nn + k));
      if (d > 0.0) {
        res.ave_ln_div[static_cast<std::size_t>(k)] += std::log(d);
        count[static_cast<std::size_t>(k)] += 1;
      }
    }
  }
  for (int k = 0; k < M; ++k) {
    if (count[static_cast<std::size_t>(k)] > 0) {
      res.ave_ln_div[static_cast<std::size_t>(k)] /=
          static_cast<double>(count[static_cast<std::size_t>(k)]);
    }
  }

  res.short_fit = fit_log_divergence(res.ave_ln_div, fs, mean_period,
                                     short_start, short_end);
  res.long_fit =
      fit_log_divergence(res.ave_ln_div, fs, mean_period, long_start, long_end);
  return res;
}

// Sample Entropy matching `chaos_algorithms.py:samp_ent_core`.
[[nodiscard]] inline double samp_ent_core(std::span<const double> data, int m,
                                          double r) {
  const int N = static_cast<int>(data.size());
  const int L = N - m;
  if (L < 2 || m < 1)
    return std::numeric_limits<double>::quiet_NaN();

  double mean = 0.0;
  for (double v : data)
    mean += v;
  mean /= static_cast<double>(N);
  double s2 = 0.0;
  for (double v : data) {
    const double d = v - mean;
    s2 += d * d;
  }
  const double R = r * std::sqrt(s2 / static_cast<double>(N - 1));
  if (R <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();

  std::int64_t count_B = 0;
  std::int64_t count_A = 0;
  for (int i = 0; i < L; ++i) {
    for (int j = i + 1; j < L; ++j) {
      bool ok_m = true;
      for (int k = 0; k < m; ++k) {
        if (std::abs(data[static_cast<std::size_t>(i + k)] -
                     data[static_cast<std::size_t>(j + k)]) > R) {
          ok_m = false;
          break;
        }
      }
      if (ok_m) {
        ++count_B;
        if (std::abs(data[static_cast<std::size_t>(i + m)] -
                     data[static_cast<std::size_t>(j + m)]) <= R) {
          ++count_A;
        }
      }
    }
  }
  if (count_A == 0 || count_B == 0)
    return std::numeric_limits<double>::quiet_NaN();
  return -std::log(static_cast<double>(count_A) / static_cast<double>(count_B));
}

// Correlation Dimension matching `chaos_algorithms.py:corrdim_core`.
[[nodiscard]] inline double corrdim_core(std::span<const double> x, int tau,
                                         int de) {
  const int N = static_cast<int>(x.size());
  const int n = N - (de - 1) * tau;
  const int k = de * tau;
  if (n <= k + 1 || de < 1 || tau < 1)
    return 0.0;

  const int n_rows = n - k - 1;
  std::vector<double> flat;
  flat.reserve(static_cast<std::size_t>(n_rows) *
               static_cast<std::size_t>(n_rows));

  for (int r1 = k + 1; r1 < n; ++r1) {
    for (int r2 = 0; r2 < n - k - 1; ++r2) {
      double s = 0.0;
      for (int c = 0; c < de; ++c) {
        const double d = x[static_cast<std::size_t>(r1 + c * tau)] -
                         x[static_cast<std::size_t>(r2 + c * tau)];
        s += d * d;
      }
      flat.push_back(std::sqrt(s));
    }
  }
  if (flat.empty())
    return 0.0;

  std::ranges::sort(flat);
  const double eps1 = flat.back();
  if (!std::isfinite(eps1) || eps1 <= 0.0)
    return 0.0;

  auto it_pos = std::ranges::upper_bound(flat, 0.0);
  if (it_pos == flat.end())
    return 0.0;
  const double eps2 = *it_pos;

  constexpr int kBins = 200;
  const double log_e2 = std::log(eps2);
  const double log_e1 = std::log(eps1);
  const double denom = static_cast<double>(n - k) * static_cast<double>(n - k);
  const double log_nk1 = std::log(static_cast<double>(n - k - 1));

  std::vector<double> eps(kBins);
  std::vector<double> CI(kBins, std::numeric_limits<double>::quiet_NaN());
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();

  for (int b = 0; b < kBins; ++b) {
    eps[static_cast<std::size_t>(b)] =
        log_e2 + (log_e1 - log_e2) * static_cast<double>(b) /
                     static_cast<double>(kBins - 1);
    const double eb = std::exp(eps[static_cast<std::size_t>(b)]);
    const auto ub = std::ranges::upper_bound(flat, eb);
    const auto cum = static_cast<double>(std::distance(flat.begin(), ub));
    if (cum > 0.0) {
      const double ci = std::log(cum / denom) + log_nk1;
      CI[static_cast<std::size_t>(b)] = ci;
      lo = std::min(lo, ci);
      hi = std::max(hi, ci);
    }
  }
  if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo)
    return 0.0;

  const double mid = 0.5 * (lo + hi);
  const double q = 0.25 * (hi - lo);

  double mx = 0.0, my = 0.0;
  int cnt = 0;
  for (int b = 0; b < kBins; ++b) {
    const double ci = CI[static_cast<std::size_t>(b)];
    if (std::isfinite(ci) && ci > mid && ci < mid + q) {
      mx += eps[static_cast<std::size_t>(b)];
      my += ci;
      ++cnt;
    }
  }
  if (cnt < 2)
    return 0.0;
  mx /= static_cast<double>(cnt);
  my /= static_cast<double>(cnt);

  double sxx = 0.0, sxy = 0.0;
  for (int b = 0; b < kBins; ++b) {
    const double ci = CI[static_cast<std::size_t>(b)];
    if (std::isfinite(ci) && ci > mid && ci < mid + q) {
      const double dx = eps[static_cast<std::size_t>(b)] - mx;
      const double dy = ci - my;
      sxx += dx * dx;
      sxy += dx * dy;
    }
  }
  return (sxx > 0.0) ? (sxy / sxx) : 0.0;
}

// Extracts all chaotic/Lyapunov features for a single preprocessed window.
[[nodiscard]] inline WindowFeatures
extract_window_features(std::span<const double> x_filt, double fs,
                        double start_mjd, int tau = 5, int dim = 5,
                        int evolve = 5) {
  WindowFeatures wf;
  wf.start_mjd = start_mjd;
  if (x_filt.size() < 200)
    return wf;

  double mean = 0.0;
  for (double v : x_filt)
    mean += v;
  mean /= static_cast<double>(x_filt.size());
  double s2 = 0.0;
  for (double v : x_filt) {
    const double d = v - mean;
    s2 += d * d;
  }
  wf.raw_std = std::sqrt(s2 / static_cast<double>(x_filt.size() - 1));
  if (wf.raw_std <= 1e-20)
    return wf;

  wf.mean_period_s = dsp::spectral_mean_period(x_filt, fs);
  const auto x_norm = zscore_normalize(x_filt);

  wf.wolf_lye = wolf_lye_core(x_norm, fs, tau, dim, evolve);
  const auto ros = rosenstein_lye_core(x_norm, fs, tau, dim, wf.mean_period_s,
                                       0.0, 1.0, 1.0, 4.0);
  wf.ros_short = ros.short_fit.slope;
  wf.ros_short_r2 = ros.short_fit.r2;
  wf.ros_long = ros.long_fit.slope;
  wf.ros_long_r2 = ros.long_fit.r2;

  if (x_norm.size() > 600) {
    const std::size_t step = x_norm.size() / 600;
    std::vector<double> x_sub;
    x_sub.reserve(600);
    for (std::size_t i = 0; i < x_norm.size(); i += step)
      x_sub.push_back(x_norm[i]);
    const int sub_tau = std::max(1, tau / static_cast<int>(step));
    wf.samp_ent = samp_ent_core(x_sub, 3, 0.2);
    wf.corr_dim = corrdim_core(x_sub, sub_tau, dim);
  } else {
    wf.samp_ent = samp_ent_core(x_norm, 3, 0.2);
    wf.corr_dim = corrdim_core(x_norm, tau, dim);
  }
  return wf;
}

} // namespace chaosmap::chaos
