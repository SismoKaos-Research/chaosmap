#pragma once

// Dense linear algebra primitives backed by LAPACK (`dgelsd_`, `dsyev_`) plus
// closed-form 2x2 and 3x3 symmetric eigensolvers for grid-level strain, FTLE,
// and 3D hypocentre PCA.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

extern "C" {
void dgelsd_(const int *m, const int *n, const int *nrhs, double *a,
             const int *lda, double *b, const int *ldb, double *s,
             const double *rcond, int *rank, double *work, const int *lwork,
             int *iwork, int *info);

void dsyev_(const char *jobz, const char *uplo, const int *n, double *a,
            const int *lda, double *w, double *work, const int *lwork,
            int *info);

void dgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             double *a, const int *lda, double *s, double *u, const int *ldu,
             double *vt, const int *ldvt, double *work, const int *lwork,
             int *info);
}

namespace chaosmap::linalg {

struct Vec2 {
  double x{};
  double y{};

  constexpr Vec2 operator+(const Vec2 &o) const noexcept {
    return {x + o.x, y + o.y};
  }
  constexpr Vec2 operator-(const Vec2 &o) const noexcept {
    return {x - o.x, y - o.y};
  }
  constexpr Vec2 operator*(double s) const noexcept { return {x * s, y * s}; }
  [[nodiscard]] constexpr double dot(const Vec2 &o) const noexcept {
    return x * o.x + y * o.y;
  }
  [[nodiscard]] double norm() const noexcept { return std::hypot(x, y); }
};

struct Vec3 {
  double x{};
  double y{};
  double z{};

  constexpr Vec3 operator+(const Vec3 &o) const noexcept {
    return {x + o.x, y + o.y, z + o.z};
  }
  constexpr Vec3 operator-(const Vec3 &o) const noexcept {
    return {x - o.x, y - o.y, z - o.z};
  }
  constexpr Vec3 operator*(double s) const noexcept {
    return {x * s, y * s, z * s};
  }
  [[nodiscard]] constexpr double dot(const Vec3 &o) const noexcept {
    return x * o.x + y * o.y + z * o.z;
  }
  [[nodiscard]] constexpr Vec3 cross(const Vec3 &o) const noexcept {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  [[nodiscard]] double norm() const noexcept { return std::hypot(x, y, z); }
  [[nodiscard]] Vec3 normalized() const noexcept {
    const double n = norm();
    return n > 0.0 ? (*this) * (1.0 / n) : Vec3{};
  }
};

// Column-major dense matrix (compatible with Fortran LAPACK routines).
class Mat {
public:
  Mat() = default;
  Mat(std::size_t rows, std::size_t cols, double init = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, init) {}

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] double *data() noexcept { return data_.data(); }
  [[nodiscard]] const double *data() const noexcept { return data_.data(); }

  [[nodiscard]] double &operator()(std::size_t r, std::size_t c) noexcept {
    assert(r < rows_ && c < cols_);
    return data_[c * rows_ + r];
  }
  [[nodiscard]] double operator()(std::size_t r, std::size_t c) const noexcept {
    assert(r < rows_ && c < cols_);
    return data_[c * rows_ + r];
  }

  [[nodiscard]] static Mat identity(std::size_t n) {
    Mat I(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      I(i, i) = 1.0;
    return I;
  }

  [[nodiscard]] Mat transpose() const {
    Mat t(cols_, rows_);
    for (std::size_t c = 0; c < cols_; ++c)
      for (std::size_t r = 0; r < rows_; ++r)
        t(c, r) = (*this)(r, c);
    return t;
  }

  [[nodiscard]] std::vector<double>
  mul_vec(std::span<const double> x) const {
    assert(x.size() == cols_);
    std::vector<double> y(rows_, 0.0);
    for (std::size_t c = 0; c < cols_; ++c) {
      const double xc = x[c];
      for (std::size_t r = 0; r < rows_; ++r)
        y[r] += (*this)(r, c) * xc;
    }
    return y;
  }

  [[nodiscard]] Mat mul(const Mat &B) const {
    assert(cols_ == B.rows_);
    Mat C(rows_, B.cols_, 0.0);
    for (std::size_t j = 0; j < B.cols_; ++j) {
      for (std::size_t k = 0; k < cols_; ++k) {
        const double bkj = B(k, j);
        for (std::size_t i = 0; i < rows_; ++i)
          C(i, j) += (*this)(i, k) * bkj;
      }
    }
    return C;
  }

private:
  std::size_t rows_{0};
  std::size_t cols_{0};
  std::vector<double> data_;
};

struct LsqResult {
  std::vector<double> x;
  Mat cov;            // (n x n) unscaled covariance (A^T W A)^+
  double rss{0.0};    // weighted residual sum of squares
  int dof{0};         // m - rank
  int rank{0};
};

// Solves min ||A x - b||_2 via SVD (`dgelsd_`).
[[nodiscard]] inline std::vector<double>
solve_lstsq(Mat A, std::span<const double> b, double rcond = 1e-10) {
  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(A.cols());
  if (m == 0 || n == 0 || static_cast<int>(b.size()) != m)
    return std::vector<double>(n, 0.0);

  const int nrhs = 1;
  const int lda = std::max(1, m);
  const int ldb = std::max({1, m, n});
  std::vector<double> rhs(static_cast<std::size_t>(ldb), 0.0);
  std::copy(b.begin(), b.end(), rhs.begin());
  std::vector<double> s(static_cast<std::size_t>(std::min(m, n)), 0.0);
  int rank = 0;
  int info = 0;

  double wkopt = 0.0;
  int iwkopt = 0;
  int lwork = -1;
  dgelsd_(&m, &n, &nrhs, A.data(), &lda, rhs.data(), &ldb, s.data(), &rcond,
          &rank, &wkopt, &lwork, &iwkopt, &info);
  lwork = std::max(1, static_cast<int>(wkopt));
  std::vector<double> work(static_cast<std::size_t>(lwork));
  std::vector<int> iwork(static_cast<std::size_t>( std::max(1, iwkopt)));

  dgelsd_(&m, &n, &nrhs, A.data(), &lda, rhs.data(), &ldb, s.data(), &rcond,
          &rank, work.data(), &lwork, iwork.data(), &info);
  rhs.resize(static_cast<std::size_t>(n));
  return rhs;
}

// Weighted linear least squares min ||sqrt(w) (A x - b)||_2^2 with full
// pseudoinverse covariance via SVD (`dgesvd_`).
// `sigma` holds the 1-sigma measurement uncertainties per row (w_i = 1/sigma_i^2).
[[nodiscard]] inline LsqResult
solve_wls(const Mat &A, std::span<const double> b,
          std::span<const double> sigma, double rcond = 1e-10) {
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  LsqResult res;
  res.x.assign(n, 0.0);
  res.cov = Mat(n, n, 0.0);
  if (m == 0 || n == 0 || b.size() != m)
    return res;

  Mat Aw(m, n);
  std::vector<double> bw(m);
  for (std::size_t i = 0; i < m; ++i) {
    const double s = (!sigma.empty() && sigma[i] > 0.0) ? sigma[i] : 1.0;
    const double w = 1.0 / s;
    bw[i] = b[i] * w;
    for (std::size_t j = 0; j < n; ++j)
      Aw(i, j) = A(i, j) * w;
  }

  const int mi = static_cast<int>(m);
  const int ni = static_cast<int>(n);
  const int k = std::min(mi, ni);
  Mat U(m, static_cast<std::size_t>(k));
  Mat VT(static_cast<std::size_t>(k), n);
  std::vector<double> S(static_cast<std::size_t>(k), 0.0);

  int lda = std::max(1, mi);
  int ldu = std::max(1, mi);
  int ldvt = std::max(1, k);
  int info = 0;
  double wkopt = 0.0;
  int lwork = -1;
  Mat Aw_copy = Aw;
  dgesvd_("S", "S", &mi, &ni, Aw_copy.data(), &lda, S.data(), U.data(), &ldu,
          VT.data(), &ldvt, &wkopt, &lwork, &info);
  lwork = std::max(1, static_cast<int>(wkopt));
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgesvd_("S", "S", &mi, &ni, Aw_copy.data(), &lda, S.data(), U.data(), &ldu,
          VT.data(), &ldvt, work.data(), &lwork, &info);

  const double thresh = S.empty() ? 0.0 : rcond * S[0];
  int rank = 0;
  std::vector<double> inv_s(static_cast<std::size_t>(k), 0.0);
  for (int r = 0; r < k; ++r) {
    if (S[static_cast<std::size_t>(r)] > thresh) {
      inv_s[static_cast<std::size_t>(r)] =
          1.0 / S[static_cast<std::size_t>(r)];
      ++rank;
    }
  }
  res.rank = rank;
  res.dof = std::max(0, mi - rank);

  // x = V * diag(inv_s) * U^T * bw
  for (int r = 0; r < k; ++r) {
    if (inv_s[static_cast<std::size_t>(r)] == 0.0)
      continue;
    double ut_b = 0.0;
    for (std::size_t i = 0; i < m; ++i)
      ut_b += U(i, static_cast<std::size_t>(r)) * bw[i];
    const double coeff = ut_b * inv_s[static_cast<std::size_t>(r)];
    for (std::size_t j = 0; j < n; ++j)
      res.x[j] += VT(static_cast<std::size_t>(r), j) * coeff;
  }

  // cov = V * diag(inv_s^2) * V^T
  for (int r = 0; r < k; ++r) {
    const double is2 =
        inv_s[static_cast<std::size_t>(r)] * inv_s[static_cast<std::size_t>(r)];
    if (is2 == 0.0)
      continue;
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i)
        res.cov(i, j) +=
            VT(static_cast<std::size_t>(r), i) *
            VT(static_cast<std::size_t>(r), j) * is2;
  }

  const auto fit = Aw.mul_vec(res.x);
  double rss = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double d = bw[i] - fit[i];
    rss += d * d;
  }
  res.rss = rss;
  return res;
}

struct Eig2 {
  std::array<double, 2> values{}; // ascending: values[0] <= values[1]
  std::array<Vec2, 2> vectors{};
};

// Closed-form 2x2 symmetric eigendecomposition for [a00 a01; a01 a11].
[[nodiscard]] inline Eig2 sym_eig2(double a00, double a01, double a11) noexcept {
  const double tr = 0.5 * (a00 + a11);
  const double diff = 0.5 * (a00 - a11);
  const double disc = std::hypot(diff, a01);
  const double l0 = tr - disc;
  const double l1 = tr + disc;

  // Principal axis angle for l1
  const double theta = 0.5 * std::atan2(2.0 * a01, a00 - a11);
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  return Eig2{
      .values = {l0, l1},
      .vectors = {Vec2{-s, c}, Vec2{c, s}},
  };
}

struct Eig3 {
  std::array<double, 3> values{}; // ascending: values[0] <= values[1] <= values[2]
  std::array<Vec3, 3> vectors{};
};

// Symmetric 3x3 eigendecomposition via LAPACK `dsyev_` (ascending eigenvalues).
[[nodiscard]] inline Eig3 sym_eig3(const std::array<double, 9> &sym_mat) {
  std::array<double, 9> a = sym_mat;
  std::array<double, 3> w{};
  const int n = 3;
  const int lda = 3;
  std::array<double, 32> work{};
  const int lwork = static_cast<int>(work.size());
  int info = 0;
  dsyev_("V", "U", &n, a.data(), &lda, w.data(), work.data(), &lwork, &info);
  return Eig3{
      .values = w,
      .vectors =
          {
              Vec3{a[0], a[1], a[2]},
              Vec3{a[3], a[4], a[5]},
              Vec3{a[6], a[7], a[8]},
          },
  };
}

// Robust median of a vector (copies input).
[[nodiscard]] inline double median(std::vector<double> v) {
  std::erase_if(v, [](double x) { return !std::isfinite(x); });
  if (v.empty())
    return std::numeric_limits<double>::quiet_NaN();
  const std::size_t mid = v.size() / 2;
  std::ranges::nth_element(v, v.begin() + static_cast<std::ptrdiff_t>(mid));
  double m = v[mid];
  if (v.size() % 2 == 0) {
    const auto max_lo = *std::max_element(
        v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
    m = 0.5 * (m + max_lo);
  }
  return m;
}

// Empirical quantile q in [0, 1] with linear interpolation.
[[nodiscard]] inline double quantile(std::vector<double> v, double q) {
  std::erase_if(v, [](double x) { return !std::isfinite(x); });
  if (v.empty())
    return std::numeric_limits<double>::quiet_NaN();
  std::ranges::sort(v);
  if (v.size() == 1)
    return v[0];
  const double pos = std::clamp(q, 0.0, 1.0) * static_cast<double>(v.size() - 1);
  const auto idx = static_cast<std::size_t>(std::floor(pos));
  const double frac = pos - static_cast<double>(idx);
  if (idx + 1 >= v.size())
    return v.back();
  return v[idx] * (1.0 - frac) + v[idx + 1] * frac;
}

// Median Absolute Deviation (MAD) scaled by 1.4826 for Gaussian consistency.
[[nodiscard]] inline double mad_sigma(std::span<const double> v, double med) {
  std::vector<double> dev;
  dev.reserve(v.size());
  for (double x : v) {
    if (std::isfinite(x))
      dev.push_back(std::abs(x - med));
  }
  return 1.4826 * median(std::move(dev));
}

} // namespace chaosmap::linalg
