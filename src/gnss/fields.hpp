#pragma once

// 2D geodetic velocity interpolation, strain-rate tensor grid (Shen et al.
// 1996 distance-weighted least squares), and Lagrangian Finite-Time Lyapunov
// Exponent (FTLE, `L-geo`) computation on the regional grid.

#include "core/linalg.hpp"
#include "geo/projection.hpp"
#include "gnss/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace chaosmap::gnss {

struct GeodeticGridFields {
  std::vector<linalg::Vec2> vel_enu_m_yr;      // interpolated (vE, vN) per cell
  std::vector<double> max_shear_nanostrain_yr; // engineering max shear e1-e2
  std::vector<double> dilatation_nanostrain_yr;
  std::vector<double> rotation_nanorad_yr; // positive clockwise
  std::vector<double> ftle_interseismic;   // L-geo before (normalized [0, 1])
  std::vector<double> ftle_coseismic; // L-geo coseismic finite strain [0, 1]
  // Extracted principal ridge polyline of the interseismic FTLE field:
  double ridge_start_lat{};
  double ridge_start_lon{};
  double ridge_end_lat{};
  double ridge_end_lon{};
  double ridge_strike_deg{};
};

namespace detail {

struct ProjectedVelStation {
  double x_km{};
  double y_km{};
  double ve_m_yr{};
  double vn_m_yr{};
  double se_m_yr{};
  double sn_m_yr{};
  double co_e_m{};
  double co_n_m{};
  double co_se_m{};
  double co_sn_m{};
  bool has_vel{false};
  bool has_co{false};
};

// Station field fitted by `local_velocity_gradient`.
enum class StationField { kVelocity, kCoseismic };

// Evaluates the local affine model v(x) = v0 + J * (x - x0) of a station field
// around `(x0_km, y0_km)` using Gaussian distance weighting (Shen et al. 1996).
// Returns {vE, vN} and the gradient J = [dvE/dx, dvE/dy; dvN/dx, dvN/dy] with
// x, y in km. For `kVelocity` the values are in m/yr and J is in (m/yr)/km =
// 1e-3 /yr = 1e6 nanostrain/yr; for `kCoseismic` the values are in m and J is
// in m/km = 1e-3 (dimensionless strain).
[[nodiscard]] inline std::pair<linalg::Vec2, std::array<double, 4>>
local_velocity_gradient(double x0_km, double y0_km,
                        std::span<const ProjectedVelStation> sta,
                        double length_scale_km = 65.0,
                        StationField field = StationField::kVelocity) {
  const bool coseismic = (field == StationField::kCoseismic);
  auto used = [coseismic](const ProjectedVelStation &s) {
    return coseismic ? s.has_co : s.has_vel;
  };
  // Adaptive length scale: L >= 0.75 * distance to the 6th-nearest station
  std::vector<double> d2s;
  d2s.reserve(sta.size());
  for (const auto &s : sta) {
    if (!used(s))
      continue;
    const double dx = s.x_km - x0_km;
    const double dy = s.y_km - y0_km;
    d2s.push_back(dx * dx + dy * dy);
  }
  if (d2s.size() < 4)
    return {{0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};

  std::ranges::sort(d2s);
  const std::size_t k_idx = std::min<std::size_t>(5, d2s.size() - 1);
  const double L = std::max(length_scale_km, 0.75 * std::sqrt(d2s[k_idx]));
  const double inv_2L2 = 1.0 / (2.0 * L * L);

  // 3x3 normal equations for [1, dx, dy]
  linalg::Mat X(d2s.size(), 3, 0.0);
  std::vector<double> be(d2s.size(), 0.0), bn(d2s.size(), 0.0);
  std::size_t row = 0;
  for (const auto &s : sta) {
    if (!used(s))
      continue;
    const double dx = s.x_km - x0_km;
    const double dy = s.y_km - y0_km;
    const double d2 = dx * dx + dy * dy;
    const double sigma = coseismic
                             ? std::max(0.002, 0.5 * (s.co_se_m + s.co_sn_m))
                             : std::max(0.0005, 0.5 * (s.se_m_yr + s.sn_m_yr));
    const double w = std::exp(-d2 * inv_2L2) / sigma;
    X(row, 0) = w;
    X(row, 1) = w * dx;
    X(row, 2) = w * dy;
    be[row] = w * (coseismic ? s.co_e_m : s.ve_m_yr);
    bn[row] = w * (coseismic ? s.co_n_m : s.vn_m_yr);
    ++row;
  }
  const auto ce = linalg::solve_lstsq(X, be);
  const auto cn = linalg::solve_lstsq(X, bn);
  return {
      {ce[0], cn[0]},
      {ce[1], ce[2], cn[1], cn[2]},
  };
}

// Bilinear interpolation of a 2D vector field stored on `grid` at metric point
// `(x_km, y_km)`.
[[nodiscard]] inline linalg::Vec2
sample_grid_vec(const geo::Grid2D &grid, std::span<const linalg::Vec2> field,
                double x_km, double y_km) noexcept {
  const auto ll = grid.proj.to_latlon(x_km, y_km);
  const double rf = std::clamp(
      (ll.x - grid.lat_min) / grid.step_deg, 0.0,
      static_cast<double>(grid.n_lat > 1 ? grid.n_lat - 1 : 0) - 1e-6);
  const double cf = std::clamp(
      (ll.y - grid.lon_min) / grid.step_deg, 0.0,
      static_cast<double>(grid.n_lon > 1 ? grid.n_lon - 1 : 0) - 1e-6);
  const auto r0 = static_cast<std::size_t>(std::floor(rf));
  const auto c0 = static_cast<std::size_t>(std::floor(cf));
  const std::size_t r1 = std::min(grid.n_lat - 1, r0 + 1);
  const std::size_t c1 = std::min(grid.n_lon - 1, c0 + 1);
  const double tr = rf - static_cast<double>(r0);
  const double tc = cf - static_cast<double>(c0);

  const auto &v00 = field[r0 * grid.n_lon + c0];
  const auto &v01 = field[r0 * grid.n_lon + c1];
  const auto &v10 = field[r1 * grid.n_lon + c0];
  const auto &v11 = field[r1 * grid.n_lon + c1];
  const auto v0 = v00 * (1.0 - tc) + v01 * tc;
  const auto v1 = v10 * (1.0 - tc) + v11 * tc;
  return v0 * (1.0 - tr) + v1 * tr;
}

} // namespace detail

// Computes the interseismic velocity field, strain-rate tensor grid, and
// Lagrangian Finite-Time Lyapunov Exponent (FTLE, `L-geo`) field on `grid`.
[[nodiscard]] inline GeodeticGridFields
compute_strain_and_ftle(std::span<const StationMotion> stations,
                        const geo::Grid2D &grid) {
  GeodeticGridFields out;
  const std::size_t N = grid.size();
  out.vel_enu_m_yr.assign(N, {});
  out.max_shear_nanostrain_yr.assign(N, 0.0);
  out.dilatation_nanostrain_yr.assign(N, 0.0);
  out.rotation_nanorad_yr.assign(N, 0.0);
  out.ftle_interseismic.assign(N, 0.0);
  out.ftle_coseismic.assign(N, 0.0);

  std::vector<detail::ProjectedVelStation> psta;
  psta.reserve(stations.size());
  for (const auto &s : stations) {
    const auto enu = grid.proj.to_enu(s.lat_deg, s.lon_deg);
    psta.push_back(detail::ProjectedVelStation{
        .x_km = enu.x,
        .y_km = enu.y,
        .ve_m_yr = s.vel_enu_m_yr.x,
        .vn_m_yr = s.vel_enu_m_yr.y,
        .se_m_yr = s.vel_sigma_m_yr.x,
        .sn_m_yr = s.vel_sigma_m_yr.y,
        .co_e_m = s.coseismic_enu_m.x,
        .co_n_m = s.coseismic_enu_m.y,
        .co_se_m = s.coseismic_sigma_m.x,
        .co_sn_m = s.coseismic_sigma_m.y,
        .has_vel = s.has_velocity,
        .has_co = s.has_coseismic,
    });
  }

  // 1. Velocity and strain-rate tensor at each grid node
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const std::size_t idx = r * grid.n_lon + c;
      const auto pt = grid.enu_at(r, c);
      const auto [v, J] = detail::local_velocity_gradient(pt.x, pt.y, psta);
      out.vel_enu_m_yr[idx] = v;

      // J entries are in (m/yr)/km = 1e-3 /yr; multiply by 1e6 for
      // nanostrain/yr
      constexpr double kNanostrainPerJ = 1.0e6;
      const double exx = kNanostrainPerJ * J[0];
      const double exy = 0.5 * kNanostrainPerJ * (J[1] + J[2]);
      const double eyy = kNanostrainPerJ * J[3];
      // Rotation rate about the vertical axis, positive clockwise
      const double rot = 0.5 * kNanostrainPerJ * (J[1] - J[2]);

      // Maximum engineering shear strain rate (e1 - e2)
      out.max_shear_nanostrain_yr[idx] = std::hypot(exx - eyy, 2.0 * exy);
      out.dilatation_nanostrain_yr[idx] = exx + eyy;
      out.rotation_nanorad_yr[idx] = rot;
    }
  }

  // 2. Finite-Time Lyapunov Exponent (FTLE) via RK4 particle advection on the
  //    velocity field. Positions are in km and velocities in m/yr, so the
  //    advection horizon of 800 km*yr/m is T = 800 kyr: a 20 mm/yr rate
  //    displaces tracers by 16 km. The FTLE is normalized to [0, 1] below, so
  //    the 1/T factor is omitted.
  constexpr double kAdvectScale = 800.0;
  constexpr int kRkSteps = 8;
  const double dt = kAdvectScale / static_cast<double>(kRkSteps);

  auto advect = [&](double x0, double y0, double sign) {
    linalg::Vec2 x{x0, y0};
    for (int step = 0; step < kRkSteps; ++step) {
      const double h = sign * dt;
      const auto k1 = detail::sample_grid_vec(grid, out.vel_enu_m_yr, x.x, x.y);
      const auto p2 = x + k1 * (0.5 * h);
      const auto k2 =
          detail::sample_grid_vec(grid, out.vel_enu_m_yr, p2.x, p2.y);
      const auto p3 = x + k2 * (0.5 * h);
      const auto k3 =
          detail::sample_grid_vec(grid, out.vel_enu_m_yr, p3.x, p3.y);
      const auto p4 = x + k3 * h;
      const auto k4 =
          detail::sample_grid_vec(grid, out.vel_enu_m_yr, p4.x, p4.y);
      x = x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
    }
    return x;
  };

  std::vector<linalg::Vec2> flow_fwd(N), flow_bwd(N);
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const auto pt = grid.enu_at(r, c);
      flow_fwd[r * grid.n_lon + c] = advect(pt.x, pt.y, +1.0);
      flow_bwd[r * grid.n_lon + c] = advect(pt.x, pt.y, -1.0);
    }
  }

  double min_ftle = 1e30, max_ftle = -1e30;
  for (std::size_t r = 1; r + 1 < grid.n_lat; ++r) {
    for (std::size_t c = 1; c + 1 < grid.n_lon; ++c) {
      const std::size_t idx = r * grid.n_lon + c;
      // Initial separations of the column and row neighbours. The grid is
      // regular in (lat, lon), so in metric coordinates these are not exactly
      // axis-aligned; the flow-map gradient is taken relative to them.
      const auto a = grid.enu_at(r, c + 1) - grid.enu_at(r, c - 1);
      const auto b = grid.enu_at(r + 1, c) - grid.enu_at(r - 1, c);
      const double det = a.x * b.y - b.x * a.y;
      if (std::abs(det) < 1e-9)
        continue;

      auto calc_sigma = [&](const std::vector<linalg::Vec2> &flow) {
        const auto fa =
            flow[r * grid.n_lon + (c + 1)] - flow[r * grid.n_lon + (c - 1)];
        const auto fb =
            flow[(r + 1) * grid.n_lon + c] - flow[(r - 1) * grid.n_lon + c];

        // F = [fa fb] * [a b]^-1
        const double F00 = (fa.x * b.y - fb.x * a.y) / det;
        const double F01 = (fb.x * a.x - fa.x * b.x) / det;
        const double F10 = (fa.y * b.y - fb.y * a.y) / det;
        const double F11 = (fb.y * a.x - fa.y * b.x) / det;

        // Cauchy-Green tensor C = F^T F
        const double C00 = F00 * F00 + F10 * F10;
        const double C01 = F00 * F01 + F10 * F11;
        const double C11 = F01 * F01 + F11 * F11;
        const auto eig = linalg::sym_eig2(C00, C01, C11);
        const double lmax = std::max(1.0, eig.values[1]);
        return 0.5 * std::log(lmax);
      };

      // Combine forward (repelling) and backward (attracting) Lagrangian
      // coherent structure ridges
      const double val = 0.5 * (calc_sigma(flow_fwd) + calc_sigma(flow_bwd));
      out.ftle_interseismic[idx] = val;
      min_ftle = std::min(min_ftle, val);
      max_ftle = std::max(max_ftle, val);
    }
  }

  // Fill border cells from nearest interior cell and normalize FTLE to [0, 1].
  // Read from a copy: normalizing in place would normalize the interior cells
  // copied into the east and north borders twice.
  const std::vector<double> raw_ftle = out.ftle_interseismic;
  const double span = std::max(1e-9, max_ftle - min_ftle);
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const std::size_t rr = std::clamp<std::size_t>(r, 1, grid.n_lat - 2);
      const std::size_t cc = std::clamp<std::size_t>(c, 1, grid.n_lon - 2);
      const double raw = raw_ftle[rr * grid.n_lon + cc];
      out.ftle_interseismic[r * grid.n_lon + c] =
          std::clamp((raw - min_ftle) / span, 0.0, 1.0);
    }
  }

  // Extract the principal ridge axis of the cells with normalized FTLE >= 0.65
  // via weighted 2D PCA
  double w_sum = 0.0, mx = 0.0, my = 0.0;
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const double v = out.ftle_interseismic[r * grid.n_lon + c];
      if (v < 0.65)
        continue;
      const double w = std::pow(v - 0.60, 2.0);
      const auto pt = grid.enu_at(r, c);
      mx += w * pt.x;
      my += w * pt.y;
      w_sum += w;
    }
  }
  if (w_sum > 0.0) {
    mx /= w_sum;
    my /= w_sum;
    double cxx = 0.0, cxy = 0.0, cyy = 0.0;
    for (std::size_t r = 0; r < grid.n_lat; ++r) {
      for (std::size_t c = 0; c < grid.n_lon; ++c) {
        const double v = out.ftle_interseismic[r * grid.n_lon + c];
        if (v < 0.65)
          continue;
        const double w = std::pow(v - 0.60, 2.0) / w_sum;
        const auto pt = grid.enu_at(r, c);
        const double dx = pt.x - mx;
        const double dy = pt.y - my;
        cxx += w * dx * dx;
        cxy += w * dx * dy;
        cyy += w * dy * dy;
      }
    }
    const auto eig = linalg::sym_eig2(cxx, cxy, cyy);
    const auto dir = eig.vectors[1];
    const double len = 1.6 * std::sqrt(std::max(10.0, eig.values[1]));
    double strike = geo::rad2deg(std::atan2(dir.x, dir.y));
    if (strike < 0.0)
      strike += 360.0;
    out.ridge_strike_deg = strike;
    const auto p0 = grid.proj.to_latlon(mx - len * dir.x, my - len * dir.y);
    const auto p1 = grid.proj.to_latlon(mx + len * dir.x, my + len * dir.y);
    out.ridge_start_lat = p0.x;
    out.ridge_start_lon = p0.y;
    out.ridge_end_lat = p1.x;
    out.ridge_end_lon = p1.y;
  }

  // 3. Coseismic FTLE: the coseismic displacement is itself a finite-time flow
  //    map x -> x + u(x), with gradient F = I + grad(u). Its largest finite
  //    stretching 0.5 * ln(lambda_max(F^T F)) is normalized to [0, 1].
  double min_co = 1e30, max_co = -1e30;
  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const std::size_t idx = r * grid.n_lon + c;
      const auto pt = grid.enu_at(r, c);
      const auto Ju =
          detail::local_velocity_gradient(pt.x, pt.y, psta, 65.0,
                                          detail::StationField::kCoseismic)
              .second;
      // Ju is in m/km; 1e-3 converts it to dimensionless strain
      const double F00 = 1.0 + 1e-3 * Ju[0];
      const double F01 = 1e-3 * Ju[1];
      const double F10 = 1e-3 * Ju[2];
      const double F11 = 1.0 + 1e-3 * Ju[3];
      const double C00 = F00 * F00 + F10 * F10;
      const double C01 = F00 * F01 + F10 * F11;
      const double C11 = F01 * F01 + F11 * F11;
      const auto eig = linalg::sym_eig2(C00, C01, C11);
      const double val = 0.5 * std::log(std::max(1.0, eig.values[1]));
      out.ftle_coseismic[idx] = val;
      min_co = std::min(min_co, val);
      max_co = std::max(max_co, val);
    }
  }
  const double span_co = max_co - min_co;
  for (double &v : out.ftle_coseismic)
    v = (span_co > 1e-12) ? std::clamp((v - min_co) / span_co, 0.0, 1.0) : 0.0;

  return out;
}

} // namespace chaosmap::gnss
