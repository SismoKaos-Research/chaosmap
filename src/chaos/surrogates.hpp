#pragma once

// Iterative Amplitude-Adjusted Fourier Transform (IAAFT) surrogates (Schreiber
// & Schmitz 1996) and a surrogate test for nonlinear statistics. A surrogate
// keeps the window's amplitude distribution exactly and its power spectrum
// almost exactly, but randomizes the Fourier phases, so it is a realization of
// a linear Gaussian process seen through a static nonlinear sensor. A
// statistic that does not differ between a window and its surrogates (e.g. a
// finite Wolf exponent of bandpassed noise) is explained by the spectrum and
// amplitude distribution alone.

#include "dsp/filter.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace chaosmap::chaos {

// Generates IAAFT surrogates of one series, reusing its FFTW plans.
class IaaftGenerator {
public:
  explicit IaaftGenerator(std::span<const double> x)
      : n_(x.size()), n_freq_(x.size() / 2 + 1), sorted_(x.begin(), x.end()) {
    std::ranges::sort(sorted_);
    if (n_ < 4)
      return;
    buf_ = fftw_alloc_real(n_);
    spec_ = fftw_alloc_complex(n_freq_);
    {
      std::scoped_lock lk(dsp::fftw_planner_mutex());
      fwd_ = fftw_plan_dft_r2c_1d(static_cast<int>(n_), buf_, spec_,
                                  FFTW_ESTIMATE);
      bwd_ = fftw_plan_dft_c2r_1d(static_cast<int>(n_), spec_, buf_,
                                  FFTW_ESTIMATE);
    }
    std::copy(x.begin(), x.end(), buf_);
    fftw_execute(fwd_);
    target_amp_.resize(n_freq_);
    for (std::size_t k = 0; k < n_freq_; ++k)
      target_amp_[k] = std::hypot(spec_[k][0], spec_[k][1]);
  }

  IaaftGenerator(const IaaftGenerator &) = delete;
  IaaftGenerator &operator=(const IaaftGenerator &) = delete;

  ~IaaftGenerator() {
    {
      std::scoped_lock lk(dsp::fftw_planner_mutex());
      if (fwd_)
        fftw_destroy_plan(fwd_);
      if (bwd_)
        fftw_destroy_plan(bwd_);
    }
    fftw_free(buf_);
    fftw_free(spec_);
  }

  [[nodiscard]] bool valid() const noexcept { return fwd_ && bwd_; }

  // Returns one surrogate: a random shuffle of the series, then alternate
  // projections onto the target Fourier amplitudes and the target value
  // distribution (rank-order remapping) until the ranks stop changing. The
  // result is the remapped series, so its values equal the input's exactly.
  [[nodiscard]] std::vector<double> generate(std::mt19937_64 &rng,
                                             int max_iter = 200) {
    std::vector<double> r = sorted_;
    if (!valid())
      return r;
    std::ranges::shuffle(r, rng);

    std::vector<std::size_t> order(n_);
    std::vector<std::size_t> prev_order;
    for (int it = 0; it < max_iter; ++it) {
      std::copy(r.begin(), r.end(), buf_);
      fftw_execute(fwd_);
      for (std::size_t k = 0; k < n_freq_; ++k) {
        const double mag = std::hypot(spec_[k][0], spec_[k][1]);
        if (mag > 0.0) {
          const double s = target_amp_[k] / mag;
          spec_[k][0] *= s;
          spec_[k][1] *= s;
        } else {
          spec_[k][0] = target_amp_[k];
          spec_[k][1] = 0.0;
        }
      }
      fftw_execute(bwd_); // unnormalized; scale does not affect the ranks

      std::iota(order.begin(), order.end(), std::size_t{0});
      std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
        return buf_[a] < buf_[b];
      });
      for (std::size_t i = 0; i < n_; ++i)
        r[order[i]] = sorted_[i];
      if (order == prev_order)
        break;
      prev_order = order;
    }
    return r;
  }

private:
  std::size_t n_;
  std::size_t n_freq_;
  std::vector<double> sorted_;
  std::vector<double> target_amp_;
  double *buf_{nullptr};
  fftw_complex *spec_{nullptr};
  fftw_plan fwd_{nullptr};
  fftw_plan bwd_{nullptr};
};

struct SurrogateTest {
  double value{std::numeric_limits<double>::quiet_NaN()};
  double surr_mean{std::numeric_limits<double>::quiet_NaN()};
  double surr_sd{std::numeric_limits<double>::quiet_NaN()};
  // (value - surr_mean) / surr_sd: how far the window departs from what its
  // spectrum and amplitude distribution alone produce.
  double z{std::numeric_limits<double>::quiet_NaN()};
  // Fraction of surrogates below `value` (ties count 1/2), in [0, 1].
  double rank_frac{std::numeric_limits<double>::quiet_NaN()};
  int n_surr{0}; // surrogates with a finite statistic
};

// Evaluates `stat` on `x` and on `n_surr` IAAFT surrogates of `x` drawn from
// a generator seeded with `seed`, so results are reproducible per window.
template <class Stat>
[[nodiscard]] SurrogateTest surrogate_test(std::span<const double> x,
                                           int n_surr, std::uint64_t seed,
                                           Stat &&stat) {
  SurrogateTest out;
  out.value = stat(x);
  if (!std::isfinite(out.value) || n_surr < 2)
    return out;

  IaaftGenerator gen(x);
  if (!gen.valid())
    return out;
  std::mt19937_64 rng(seed);
  std::vector<double> vals;
  vals.reserve(static_cast<std::size_t>(n_surr));
  for (int k = 0; k < n_surr; ++k) {
    const auto s = gen.generate(rng);
    const double v = stat(std::span<const double>(s));
    if (std::isfinite(v))
      vals.push_back(v);
  }
  out.n_surr = static_cast<int>(vals.size());
  if (vals.size() < 2)
    return out;

  double mean = 0.0;
  for (double v : vals)
    mean += v;
  mean /= static_cast<double>(vals.size());
  double s2 = 0.0;
  double below = 0.0;
  for (double v : vals) {
    s2 += (v - mean) * (v - mean);
    below += (v < out.value) ? 1.0 : (v == out.value ? 0.5 : 0.0);
  }
  out.surr_mean = mean;
  out.surr_sd = std::sqrt(s2 / static_cast<double>(vals.size() - 1));
  out.z = (out.surr_sd > 0.0) ? (out.value - mean) / out.surr_sd
                              : std::numeric_limits<double>::quiet_NaN();
  out.rank_frac = below / static_cast<double>(vals.size());
  return out;
}

} // namespace chaosmap::chaos
