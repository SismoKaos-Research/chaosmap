#pragma once

// Signal preprocessing (demean, linear detrend, 4-pole zero-phase Butterworth
// bandpass via cascaded biquad SOS, integer decimation) and FFTW3-backed
// Hann-windowed spectral mean period matching `chaos/src/chaos/preprocess.py`
// and `chaos/src/chaos/mean_period.py`.

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <mutex>
#include <numbers>
#include <span>
#include <vector>

namespace chaosmap::dsp {

// FFTW's planner is not thread-safe: every plan creation and destruction in
// the program must hold this mutex (`fftw_execute` itself is safe).
[[nodiscard]] inline std::mutex &fftw_planner_mutex() {
  static std::mutex m;
  return m;
}

struct Biquad {
  double b0{1.0};
  double b1{0.0};
  double b2{0.0};
  double a1{0.0}; // a0 is normalized to 1.0
  double a2{0.0};
};

// Subtracts mean and least-squares linear trend in place (matching ObsPy
// `detrend('demean')` followed by `detrend('linear')`).
inline void demean_and_detrend(std::span<double> x) noexcept {
  const std::size_t n = x.size();
  if (n == 0)
    return;
  double mean = 0.0;
  for (double v : x)
    mean += v;
  mean /= static_cast<double>(n);
  for (double &v : x)
    v -= mean;
  if (n < 2)
    return;

  const double mid = 0.5 * static_cast<double>(n - 1);
  double s_xy = 0.0;
  double s_xx = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) - mid;
    s_xy += t * x[i];
    s_xx += t * t;
  }
  if (s_xx > 0.0) {
    const double slope = s_xy / s_xx;
    for (std::size_t i = 0; i < n; ++i) {
      x[i] -= slope * (static_cast<double>(i) - mid);
    }
  }
}

// Designs an N-pole (default N=4 -> 4 biquad sections = 8th-order digital
// bandpass) Butterworth bandpass filter via exact bilinear transform with
// frequency pre-warping, matching `scipy.signal.butter(order, [fmin, fmax],
// btype='band', fs=fs, output='sos')`.
[[nodiscard]] inline std::vector<Biquad>
butterworth_bandpass_sos(double fs, double fmin, double fmax, int order = 4) {
  std::vector<Biquad> sos;
  if (fs <= 0.0 || fmin <= 0.0 || fmax <= fmin || fmax >= 0.5 * fs ||
      order <= 0)
    return sos;

  // Pre-warped analog frequencies (for T = 1 -> s = 2*(1 - z^-1)/(1 + z^-1))
  const double w1 = 2.0 * std::tan(std::numbers::pi * fmin / fs);
  const double w2 = 2.0 * std::tan(std::numbers::pi * fmax / fs);
  const double bw = w2 - w1;
  const double w0_sq = w1 * w2;

  sos.reserve(static_cast<std::size_t>(order));
  const int pairs = order / 2;
  for (int k = 0; k < pairs; ++k) {
    const double theta = std::numbers::pi *
                         static_cast<double>(2 * k + order + 1) /
                         static_cast<double>(2 * order);
    const std::complex<double> pk(std::cos(theta), std::sin(theta));
    const std::complex<double> b_pk = bw * pk;
    const std::complex<double> disc = std::sqrt(b_pk * b_pk - 4.0 * w0_sq);
    const std::complex<double> s_plus = 0.5 * (b_pk + disc);
    const std::complex<double> s_minus = 0.5 * (b_pk - disc);

    const double denom_norm = std::abs(2.0 - s_plus) * std::abs(2.0 - s_minus);
    const double g = (denom_norm > 0.0) ? (2.0 * bw / denom_norm) : 0.0;

    for (const auto &sp : {s_plus, s_minus}) {
      const std::complex<double> zp = (2.0 + sp) / (2.0 - sp);
      Biquad sec;
      sec.b0 = g;
      sec.b1 = 0.0;
      sec.b2 = -g;
      sec.a1 = -2.0 * zp.real();
      sec.a2 = std::norm(zp);
      sos.push_back(sec);
    }
  }
  return sos;
}

// Applies a cascade of biquad sections in a single direction (Direct Form II
// Transposed).
inline void sosfilt_inplace(std::span<const Biquad> sos,
                            std::span<double> x) noexcept {
  for (const auto &s : sos) {
    double z1 = 0.0;
    double z2 = 0.0;
    for (double &v : x) {
      const double y = s.b0 * v + z1;
      z1 = s.b1 * v - s.a1 * y + z2;
      z2 = s.b2 * v - s.a2 * y;
      v = y;
    }
  }
}

// Zero-phase forward-backward biquad filtering (`zerophase=True` in ObsPy).
inline void sosfiltfilt_inplace(std::span<const Biquad> sos,
                                std::span<double> x) noexcept {
  if (x.size() < 4 || sos.empty())
    return;
  sosfilt_inplace(sos, x);
  std::ranges::reverse(x);
  sosfilt_inplace(sos, x);
  std::ranges::reverse(x);
}

// Preprocesses a raw waveform window: demean + linear detrend + 4-corner
// zero-phase Butterworth bandpass [fmin, fmax] + integer decimation to
// `target_fs`.
[[nodiscard]] inline std::vector<double>
preprocess_window(std::span<const double> raw, double raw_fs, double target_fs,
                  double fmin, double fmax) {
  std::vector<double> x(raw.begin(), raw.end());
  if (x.empty())
    return x;
  demean_and_detrend(x);
  const auto sos = butterworth_bandpass_sos(raw_fs, fmin, fmax, 4);
  sosfiltfilt_inplace(sos, x);

  const auto factor = static_cast<std::size_t>(
      std::max(1L, std::lround(raw_fs / std::max(1e-6, target_fs))));
  if (factor <= 1)
    return x;

  std::vector<double> dec;
  dec.reserve((x.size() + factor - 1) / factor);
  for (std::size_t i = 0; i < x.size(); i += factor) {
    dec.push_back(x[i]);
  }
  return dec;
}

// Computes the spectral mean period (1 / spectral_centroid_frequency) using a
// Hann-windowed one-sided periodogram via FFTW3, matching
// `chaos/src/chaos/mean_period.py:spectral_mean_period`.
[[nodiscard]] inline double spectral_mean_period(std::span<const double> x,
                                                 double fs) {
  const std::size_t n = x.size();
  if (n < 8 || fs <= 0.0)
    return 1.0;

  double mean = 0.0;
  for (double v : x)
    mean += v;
  mean /= static_cast<double>(n);

  std::vector<double> in(n);
  const double inv_nm1 = 1.0 / static_cast<double>(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    const double hann =
        0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * static_cast<double>(i) *
                              inv_nm1));
    in[i] = (x[i] - mean) * hann;
  }

  const std::size_t n_freq = n / 2 + 1;
  std::vector<fftw_complex> out(n_freq);

  fftw_plan plan = nullptr;
  {
    std::scoped_lock lk(fftw_planner_mutex());
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(n), in.data(), out.data(),
                                FFTW_ESTIMATE);
  }
  if (!plan)
    return 1.0;

  fftw_execute(plan);

  {
    std::scoped_lock lk(fftw_planner_mutex());
    fftw_destroy_plan(plan);
  }

  const double df = fs / static_cast<double>(n);
  double num = 0.0;
  double den = 0.0;
  // Skip k = 0 (DC component), matching `freqs = freqs[1:]`, `pxx = pxx[1:]`
  for (std::size_t k = 1; k < n_freq; ++k) {
    const double re = out[k][0];
    const double im = out[k][1];
    double pxx = re * re + im * im;
    // One-sided periodogram doubles interior bins (1 .. n_freq - 2 when n even)
    if (k + 1 < n_freq || (n % 2 != 0))
      pxx *= 2.0;
    const double fk = static_cast<double>(k) * df;
    num += fk * pxx;
    den += pxx;
  }

  if (den <= 1e-30 || num <= 1e-30)
    return 1.0;
  const double mean_freq = num / den;
  return 1.0 / mean_freq;
}

} // namespace chaosmap::dsp
