#pragma once

// Exact analytical solution for surface displacements due to a rectangular
// shear/tensile fault in an elastic half-space (Okada 1985, Bull. Seismol.
// Soc. Am., 75(4), 1135-1154, Equations 25-30).

#include "core/linalg.hpp"
#include "geo/projection.hpp"

#include <cmath>
#include <numbers>

namespace chaosmap::okada {

namespace detail {

struct SubFunctions {
  double I1{}, I2{}, I3{}, I4{}, I5{};
};

// Evaluates Okada (1985) Equations (28)-(29) at Chinnery integration corner
// (xi, eta), given p = y cos(d) + d sin(d) and q = y sin(d) - d cos(d).
// alpha = mu / (lambda + mu) = 1 - 2 * nu.
[[nodiscard]] inline SubFunctions eval_I(double xi, double eta, double q,
                                         double sin_d, double cos_d,
                                         double alpha) noexcept {
  const double R = std::hypot(xi, eta, q);
  if (R < 1e-12)
    return {};
  const double y_tilde = eta * cos_d + q * sin_d;
  const double d_tilde = eta * sin_d - q * cos_d;
  const double R_eta = R + eta;
  const double R_dt = R + d_tilde;
  const double ln_R_eta =
      (R_eta > 1e-12) ? std::log(R_eta) : -std::log(R - eta);

  SubFunctions s;
  if (std::abs(cos_d) > 1e-6) {
    // General dip (Eq. 28)
    s.I5 = (std::abs(xi) < 1e-12)
               ? 0.0
               : alpha * 2.0 / cos_d *
                     std::atan((eta * (R + q * cos_d) +
                                R * (R + std::abs(xi)) * sin_d) /
                               (xi * (R + std::abs(xi)) * cos_d));
    if (xi < 0.0) {
      // atan above uses |xi| in denominator factor (R+|xi|), xi sign already in
      // denominator
    }
    // More direct form of Eq. (28) for I5:
    const double X = std::hypot(xi, q);
    if (std::abs(xi) < 1e-12) {
      s.I5 = 0.0;
    } else {
      s.I5 = alpha * 2.0 / cos_d *
             std::atan((eta * (X + q * cos_d) + X * (R + X) * sin_d) /
                       (xi * (R + X) * cos_d));
    }
    s.I4 = alpha / cos_d * (std::log(R_dt) - sin_d * ln_R_eta);
    s.I3 = alpha * (1.0 / cos_d * y_tilde / R_dt - ln_R_eta) +
           (sin_d / cos_d) * s.I4;
    s.I2 = alpha * (-ln_R_eta) - s.I3;
    s.I1 = alpha * (-1.0 / cos_d * xi / R_dt) - (sin_d / cos_d) * s.I5;
  } else {
    // Vertical fault cos(delta) -> 0 (Eq. 29)
    const double R_dt2 = R_dt * R_dt;
    s.I1 = -0.5 * alpha * xi * q / R_dt2;
    s.I3 = 0.5 * alpha * (eta / R_dt + y_tilde * q / R_dt2 - ln_R_eta);
    s.I2 = alpha * (-ln_R_eta) - s.I3;
    s.I4 = -alpha * q * sin_d / R_dt;
    s.I5 = -alpha * xi * sin_d / R_dt;
  }
  return s;
}

// Evaluates primitive f(xi, eta) for Strike-Slip (Eq. 25), Dip-Slip (Eq. 26),
// and Tensile (Eq. 27) at a single Chinnery corner (xi, eta).
[[nodiscard]] inline linalg::Vec3
chinnery_corner(double xi, double eta, double q, double sin_d, double cos_d,
                double U1, double U2, double U3, double alpha) noexcept {
  const double R = std::hypot(xi, eta, q);
  if (R < 1e-12)
    return {};
  const double y_tilde = eta * cos_d + q * sin_d;
  const double d_tilde = eta * sin_d - q * cos_d;
  const double R_eta = std::max(1e-12, R + eta);
  const double R_xi = std::max(1e-12, R + xi);
  const double theta =
      (std::abs(q) < 1e-12) ? 0.0 : std::atan(xi * eta / (q * R));
  const auto I = eval_I(xi, eta, q, sin_d, cos_d, alpha);

  double ux = 0.0, uy = 0.0, uz = 0.0;
  constexpr double kInv2Pi = 1.0 / (2.0 * std::numbers::pi);

  // Strike-slip (Eq. 25)
  if (U1 != 0.0) {
    const double fx = xi * q / (R * R_eta) + theta + I.I1 * sin_d;
    const double fy =
        y_tilde * q / (R * R_eta) + q * cos_d / R_eta + I.I2 * sin_d;
    const double fz =
        d_tilde * q / (R * R_eta) + q * sin_d / R_eta + I.I4 * sin_d;
    ux -= U1 * kInv2Pi * fx;
    uy -= U1 * kInv2Pi * fy;
    uz -= U1 * kInv2Pi * fz;
  }

  // Dip-slip (Eq. 26)
  if (U2 != 0.0) {
    const double fx = q / R - I.I3 * sin_d * cos_d;
    const double fy =
        y_tilde * q / (R * R_xi) + cos_d * theta - I.I1 * sin_d * cos_d;
    const double fz =
        d_tilde * q / (R * R_xi) + sin_d * theta - I.I5 * sin_d * cos_d;
    ux -= U2 * kInv2Pi * fx;
    uy -= U2 * kInv2Pi * fy;
    uz -= U2 * kInv2Pi * fz;
  }

  // Tensile (Eq. 27)
  if (U3 != 0.0) {
    const double q2 = q * q;
    const double fx = q2 / (R * R_eta) - I.I3 * sin_d * sin_d;
    const double fy = -d_tilde * q / (R * R_xi) -
                      sin_d * (xi * q / (R * R_eta) - theta) -
                      I.I1 * sin_d * sin_d;
    const double fz = y_tilde * q / (R * R_xi) +
                      cos_d * (xi * q / (R * R_eta) - theta) -
                      I.I5 * sin_d * sin_d;
    ux += U3 * kInv2Pi * fx;
    uy += U3 * kInv2Pi * fy;
    uz += U3 * kInv2Pi * fz;
  }

  return {ux, uy, uz};
}

} // namespace detail

// Direct evaluation in Okada (1985) local fault coordinate system:
// x: along-strike from lower-left corner [0, L]
// y: horizontal perpendicular to strike from lower-left reference line
// d: depth of bottom edge of fault (d > 0)
// delta_deg: dip angle in degrees
// U1, U2, U3: strike-slip (left-lateral > 0), dip-slip (thrust > 0), tensile
// nu: Poisson's ratio (0.25 -> lambda == mu, alpha = 0.5)
[[nodiscard]] inline linalg::Vec3 okada85_local(double x, double y, double d,
                                                double delta_deg, double L,
                                                double W, double U1, double U2,
                                                double U3 = 0.0,
                                                double nu = 0.25) noexcept {
  const double d_rad = geo::deg2rad(delta_deg);
  const double sin_d = std::sin(d_rad);
  const double cos_d = std::cos(d_rad);
  const double p = y * cos_d + d * sin_d;
  const double q = y * sin_d - d * cos_d;
  const double alpha = (1.0 - 2.0 * nu); // mu / (lambda + mu) = (1 - 2*nu)

  // Chinnery's notation (Eq. 24):
  // f(xi, eta)|| = f(x, p) - f(x, p - W) - f(x - L, p) + f(x - L, p - W)
  const auto f11 =
      detail::chinnery_corner(x, p, q, sin_d, cos_d, U1, U2, U3, alpha);
  const auto f12 =
      detail::chinnery_corner(x, p - W, q, sin_d, cos_d, U1, U2, U3, alpha);
  const auto f21 =
      detail::chinnery_corner(x - L, p, q, sin_d, cos_d, U1, U2, U3, alpha);
  const auto f22 =
      detail::chinnery_corner(x - L, p - W, q, sin_d, cos_d, U1, U2, U3, alpha);

  return (f11 - f12) - (f21 - f22);
}

// Rectangular fault patch in metric ENU coordinates (km) with slip in metres.
// Convention (standard Aki & Richards / Okada):
// - `strike_deg`: azimuth clockwise from North (0..360); fault dips to the
//   RIGHT when looking along strike.
// - `dip_deg`: dip angle downward from horizontal (0..90).
// - `(east_km, north_km, depth_km)`: centroid of the rectangular patch.
// - `slip_strike_m`: strike-slip component (positive = left-lateral; for
//   right-lateral slip use negative, or convert from `rake_deg` where
//   `U1 = slip * cos(rake)`, `U2 = slip * sin(rake)`).
// - `slip_dip_m`: dip-slip component (positive = reverse/thrust).
struct FaultPatch {
  double east_km{};
  double north_km{};
  double depth_km{10.0}; // centroid depth > 0
  double length_km{30.0};
  double width_km{15.0};
  double strike_deg{245.0};
  double dip_deg{50.0};
  double slip_strike_m{0.0}; // U1 (left-lateral > 0)
  double slip_dip_m{0.0};    // U2 (thrust > 0)
};

// Computes surface displacement (East, North, Up in metres) at observation
// point `(obs_east_km, obs_north_km)` due to `patch`.
[[nodiscard]] inline linalg::Vec3 dislocation_enu(const FaultPatch &patch,
                                                  double obs_east_km,
                                                  double obs_north_km,
                                                  double nu = 0.25) noexcept {
  const double str_rad = geo::deg2rad(patch.strike_deg);
  const double dip_rad = geo::deg2rad(patch.dip_deg);
  const double sin_s = std::sin(str_rad);
  const double cos_s = std::cos(str_rad);
  const double sin_d = std::sin(dip_rad);
  const double cos_d = std::cos(dip_rad);

  // Unit vectors in (East, North):
  // Along-strike (Okada +x): (sin_s, cos_s)
  // Up-dip horizontal projection (Okada +y, to the LEFT of strike since fault
  // dips to the right of strike): (-cos_s, sin_s)
  const double dx = obs_east_km - patch.east_km;
  const double dy = obs_north_km - patch.north_km;

  const double x_c = dx * sin_s + dy * cos_s;
  const double y_c = -dx * cos_s + dy * sin_s;

  // Shift from patch centroid to Okada's lower-left reference corner (0, 0, -d)
  const double x_ok = x_c + 0.5 * patch.length_km;
  const double y_ok = y_c + 0.5 * patch.width_km * cos_d;
  const double d_bottom =
      std::max(0.2, patch.depth_km + 0.5 * patch.width_km * sin_d);

  const auto u_loc = okada85_local(
      x_ok, y_ok, d_bottom, patch.dip_deg, patch.length_km, patch.width_km,
      patch.slip_strike_m, patch.slip_dip_m, 0.0, nu);

  // Rotate (ux, uy, uz) back to (East, North, Up):
  const double u_east = u_loc.x * sin_s - u_loc.y * cos_s;
  const double u_north = u_loc.x * cos_s + u_loc.y * sin_s;
  return {u_east, u_north, u_loc.z};
}

} // namespace chaosmap::okada
