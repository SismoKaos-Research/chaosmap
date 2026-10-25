#pragma once

// Geodesic distance/azimuth and local tangent-plane (ENU) projection for
// converting between geographic coordinates (lat, lon in degrees) and metric
// fault/grid coordinates (east_km, north_km).

#include "core/linalg.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace chaosmap::geo {

inline constexpr double kEarthRadiusKm = 6371.0088;
inline constexpr double kDegToRad = std::numbers::pi / 180.0;
inline constexpr double kRadToDeg = 180.0 / std::numbers::pi;

[[nodiscard]] constexpr double deg2rad(double d) noexcept {
  return d * kDegToRad;
}
[[nodiscard]] constexpr double rad2deg(double r) noexcept {
  return r * kRadToDeg;
}

[[nodiscard]] constexpr double wrap_lon(double lon_deg) noexcept {
  while (lon_deg > 180.0)
    lon_deg -= 360.0;
  while (lon_deg < -180.0)
    lon_deg += 360.0;
  return lon_deg;
}

// Great-circle distance in km via the Haversine formula.
[[nodiscard]] inline double haversine_km(double lat1_deg, double lon1_deg,
                                         double lat2_deg,
                                         double lon2_deg) noexcept {
  const double p1 = deg2rad(lat1_deg);
  const double p2 = deg2rad(lat2_deg);
  const double dp = p2 - p1;
  const double dl = deg2rad(lon2_deg - lon1_deg);
  const double s1 = std::sin(0.5 * dp);
  const double s2 = std::sin(0.5 * dl);
  const double a =
      s1 * s1 + std::cos(p1) * std::cos(p2) * s2 * s2;
  return 2.0 * kEarthRadiusKm * std::asin(std::sqrt(std::clamp(a, 0.0, 1.0)));
}

// Local azimuthal-equidistant / tangent-plane projection centered at
// (lat0_deg, lon0_deg), returning (east_km, north_km) accurate to <0.05% across
// 300 km regional boxes.
class LocalProjection {
public:
  constexpr LocalProjection() = default;
  explicit LocalProjection(double lat0_deg, double lon0_deg) noexcept
      : lat0_deg_(lat0_deg), lon0_deg_(wrap_lon(lon0_deg)),
        cos_lat0_(std::cos(deg2rad(lat0_deg))) {}

  [[nodiscard]] constexpr double lat0() const noexcept { return lat0_deg_; }
  [[nodiscard]] constexpr double lon0() const noexcept { return lon0_deg_; }

  // Projects (lat_deg, lon_deg) -> (east_km, north_km).
  [[nodiscard]] linalg::Vec2 to_enu(double lat_deg,
                                    double lon_deg) const noexcept {
    const double dlat = deg2rad(lat_deg - lat0_deg_);
    const double dlon = deg2rad(wrap_lon(lon_deg - lon0_deg_));
    const double mid_cos = std::cos(deg2rad(0.5 * (lat_deg + lat0_deg_)));
    return {
        kEarthRadiusKm * dlon * mid_cos,
        kEarthRadiusKm * dlat,
    };
  }

  // Inverse: (east_km, north_km) -> (lat_deg, lon_deg).
  [[nodiscard]] linalg::Vec2 to_latlon(double east_km,
                                       double north_km) const noexcept {
    const double lat = lat0_deg_ + rad2deg(north_km / kEarthRadiusKm);
    const double mid_cos =
        std::max(1e-6, std::cos(deg2rad(0.5 * (lat + lat0_deg_))));
    const double lon =
        wrap_lon(lon0_deg_ + rad2deg(east_km / (kEarthRadiusKm * mid_cos)));
    return {lat, lon};
  }

  [[nodiscard]] constexpr double cos_lat0() const noexcept { return cos_lat0_; }

private:
  double lat0_deg_{39.0};
  double lon0_deg_{35.0};
  double cos_lat0_{0.77714596145};
};

// Regular 2D geographic/metric grid over [lat_min, lat_max] x [lon_min, lon_max].
struct Grid2D {
  LocalProjection proj;
  double lat_min{};
  double lat_max{};
  double lon_min{};
  double lon_max{};
  double step_deg{0.05};
  std::size_t n_lat{0};
  std::size_t n_lon{0};

  static Grid2D make(double lat_min, double lat_max, double lon_min,
                     double lon_max, double step_deg = 0.05) {
    Grid2D g;
    g.lat_min = lat_min;
    g.lat_max = lat_max;
    g.lon_min = lon_min;
    g.lon_max = lon_max;
    g.step_deg = step_deg;
    g.proj = LocalProjection(0.5 * (lat_min + lat_max),
                             0.5 * (lon_min + lon_max));
    g.n_lat = static_cast<std::size_t>(
                  std::floor((lat_max - lat_min) / step_deg + 1e-9)) +
              1;
    g.n_lon = static_cast<std::size_t>(
                  std::floor((lon_max - lon_min) / step_deg + 1e-9)) +
              1;
    return g;
  }

  [[nodiscard]] double lat_at(std::size_t r) const noexcept {
    return lat_min + static_cast<double>(r) * step_deg;
  }
  [[nodiscard]] double lon_at(std::size_t c) const noexcept {
    return lon_min + static_cast<double>(c) * step_deg;
  }
  [[nodiscard]] linalg::Vec2 enu_at(std::size_t r,
                                    std::size_t c) const noexcept {
    return proj.to_enu(lat_at(r), lon_at(c));
  }
  [[nodiscard]] std::size_t size() const noexcept { return n_lat * n_lon; }
};

} // namespace chaosmap::geo
