#pragma once

// 3D hypocentre plane fitting (PCA / OADC-style anisotropic clustering) for
// extracting pre-event background fault planes and post-event aftershock
// rupture planes.

#include "core/linalg.hpp"
#include "geo/projection.hpp"
#include "io/catalog.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

namespace chaosmap::seis {

struct FittedPlane {
  double center_lat{};
  double center_lon{};
  double center_depth_km{};
  double strike_deg{}; // 0..360, dipping to the right of strike
  double dip_deg{};    // 0..90
  double length_km{};
  double width_km{};
  double thickness_km{}; // 1-sigma fault-normal thickness
  double planarity{};    // 1 - lambda_0 / lambda_1 in [0, 1]
  double trace_start_lat{};
  double trace_start_lon{};
  double trace_end_lat{};
  double trace_end_lon{};
  std::size_t n_events{0};
};

struct SeismicityAnalysis {
  FittedPlane dominant_plane;
  std::vector<FittedPlane> segments; // along-strike OADC sub-planes
  std::vector<double> grid_density;  // normalized [0, 1] on Grid2D
  std::vector<double> grid_planarity;
};

// Fits a 3D fault plane to a cloud of earthquake hypocentres using weighted
// 3D PCA (weighted by M^2 so larger shocks anchor the plane).
[[nodiscard]] inline FittedPlane
fit_hypocentre_plane(std::span<const io::CatalogEvent> events,
                     const geo::LocalProjection &proj) {
  FittedPlane out;
  out.n_events = events.size();
  if (events.size() < 4)
    return out;

  std::vector<linalg::Vec3> pts(events.size());
  std::vector<double> w(events.size(), 1.0);
  double w_sum = 0.0;
  linalg::Vec3 mean{};

  for (std::size_t i = 0; i < events.size(); ++i) {
    const auto enu = proj.to_enu(events[i].lat_deg, events[i].lon_deg);
    // Coordinates in km: x = East, y = North, z = Up = -depth_km
    pts[i] = {enu.x, enu.y, -events[i].depth_km};
    const double wi = std::pow(std::max(1.0, events[i].magnitude), 2.0);
    w[i] = wi;
    w_sum += wi;
    mean = mean + pts[i] * wi;
  }
  mean = mean * (1.0 / w_sum);

  // Stage 1: 2D horizontal weighted PCA (East-North) for fault strike and length
  double cxx = 0.0, cxy = 0.0, cyy = 0.0;
  for (std::size_t i = 0; i < events.size(); ++i) {
    const double dx = pts[i].x - mean.x;
    const double dy = pts[i].y - mean.y;
    const double wi = w[i] / w_sum;
    cxx += wi * dx * dx;
    cxy += wi * dx * dy;
    cyy += wi * dy * dy;
  }
  const auto eig_xy = linalg::sym_eig2(cxx, cxy, cyy);
  // eig_xy.values[1] >= eig_xy.values[0]; vectors[1] is along-strike unit vector
  const linalg::Vec2 major_xy = eig_xy.vectors[1];
  double strike_deg = geo::rad2deg(std::atan2(major_xy.x, major_xy.y));
  if (strike_deg < 0.0)
    strike_deg += 360.0;

  // Stage 2: Vertical cross-section perpendicular to strike for fault dip
  // Right-hand perpendicular horizontal vector (90 deg clockwise from strike):
  const double s_rad0 = geo::deg2rad(strike_deg);
  const double perp_x = std::cos(s_rad0);
  const double perp_y = -std::sin(s_rad0);
  const double sigma_perp =
      std::clamp(std::sqrt(std::max(1.0, eig_xy.values[0])), 4.0, 12.0);

  double cuu = 0.0, cuv = 0.0, cvv = 0.0, w_cs = 0.0;
  for (std::size_t i = 0; i < events.size(); ++i) {
    const double dx = pts[i].x - mean.x;
    const double dy = pts[i].y - mean.y;
    const double u = dx * perp_x + dy * perp_y; // across-strike (+ to right, km)
    const double v = (-pts[i].z) - (-mean.z);   // depth (+ downward, km)
    if (std::abs(u) > 1.25 * sigma_perp)
      continue;
    const double wi = w[i];
    cuu += wi * u * u;
    cuv += wi * u * v;
    cvv += wi * v * v;
    w_cs += wi;
  }
  if (w_cs > 0.0) {
    cuu /= w_cs;
    cuv /= w_cs;
    cvv /= w_cs;
  }

  // If cuv < 0, depth increases to the left of strike_deg, so flip strike by 180 deg
  // to obey the right-hand rule (dip to the right of strike).
  if (cuv < 0.0) {
    strike_deg = std::fmod(strike_deg + 180.0, 360.0);
    cuv = -cuv;
  }
  out.strike_deg = strike_deg;

  // Physical (km, km) 2D orthogonal regression in the core cross-section:
  const auto eig_uv = linalg::sym_eig2(cuu, cuv, cvv);
  const double du = std::abs(eig_uv.vectors[1].x);
  const double dv = std::abs(eig_uv.vectors[1].y);
  out.dip_deg = std::clamp(geo::rad2deg(std::atan2(dv, std::max(1e-6, du))),
                           20.0, 89.0);

  out.thickness_km = std::sqrt(std::max(0.0, eig_uv.values[0]));
  out.width_km =
      std::clamp(3.2 * std::sqrt(std::max(0.0, eig_uv.values[1])), 8.0, 35.0);
  out.length_km =
      std::clamp(3.2 * std::sqrt(std::max(0.0, eig_xy.values[1])), 10.0, 250.0);
  out.planarity =
      (eig_xy.values[1] > 1e-6)
          ? std::clamp(1.0 - eig_xy.values[0] / eig_xy.values[1], 0.0, 1.0)
          : 0.0;

  const auto c_ll = proj.to_latlon(mean.x, mean.y);
  out.center_lat = c_ll.x;
  out.center_lon = c_ll.y;
  out.center_depth_km = -mean.z;

  const double str_rad = geo::deg2rad(out.strike_deg);
  const double sin_s = std::sin(str_rad);
  const double cos_s = std::cos(str_rad);
  const auto tr0 = proj.to_latlon(mean.x - 0.5 * out.length_km * sin_s,
                                  mean.y - 0.5 * out.length_km * cos_s);
  const auto tr1 = proj.to_latlon(mean.x + 0.5 * out.length_km * sin_s,
                                  mean.y + 0.5 * out.length_km * cos_s);
  out.trace_start_lat = tr0.x;
  out.trace_start_lon = tr0.y;
  out.trace_end_lat = tr1.x;
  out.trace_end_lon = tr1.y;
  return out;
}

// Performs regional seismicity analysis (dominant plane, along-strike sub-planes,
// and gridded density/planarity on `grid`).
[[nodiscard]] inline SeismicityAnalysis
analyze_seismicity(std::span<const io::CatalogEvent> events,
                   const geo::Grid2D &grid, std::size_t n_segments = 3) {
  SeismicityAnalysis out;
  out.dominant_plane = fit_hypocentre_plane(events, grid.proj);
  out.grid_density.assign(grid.size(), 0.0);
  out.grid_planarity.assign(grid.size(), 0.0);
  if (events.empty())
    return out;

  // Partition events along the dominant strike axis into `n_segments` OADC
  // sub-segments to capture fault bends/splays.
  const double str_rad = geo::deg2rad(out.dominant_plane.strike_deg);
  const double sin_s = std::sin(str_rad);
  const double cos_s = std::cos(str_rad);
  const auto c0 = grid.proj.to_enu(out.dominant_plane.center_lat,
                                   out.dominant_plane.center_lon);

  struct IndexedEvent {
    double along_km{};
    io::CatalogEvent ev;
  };
  std::vector<IndexedEvent> sorted;
  sorted.reserve(events.size());
  for (const auto &e : events) {
    const auto p = grid.proj.to_enu(e.lat_deg, e.lon_deg);
    const double s = (p.x - c0.x) * sin_s + (p.y - c0.y) * cos_s;
    sorted.push_back({s, e});
  }
  std::ranges::sort(sorted, {}, &IndexedEvent::along_km);

  if (sorted.size() >= 12 && n_segments >= 2) {
    for (std::size_t k = 0; k < n_segments; ++k) {
      const std::size_t i0 = (k * sorted.size()) / n_segments;
      const std::size_t i1 = ((k + 1) * sorted.size()) / n_segments;
      std::vector<io::CatalogEvent> sub;
      sub.reserve(i1 - i0);
      for (std::size_t i = i0; i < i1; ++i)
        sub.push_back(sorted[i].ev);
      out.segments.push_back(fit_hypocentre_plane(sub, grid.proj));
    }
  } else if (out.dominant_plane.n_events >= 4) {
    out.segments.push_back(out.dominant_plane);
  }

  // Smooth Gaussian kernel density on `grid` (bandwidth = 12 km)
  double max_d = 0.0;
  constexpr double kSigmaKm = 12.0;
  constexpr double kInv2Sig2 = 1.0 / (2.0 * kSigmaKm * kSigmaKm);
  std::vector<linalg::Vec2> ev_enu(events.size());
  std::vector<double> ev_w(events.size());
  for (std::size_t i = 0; i < events.size(); ++i) {
    ev_enu[i] = grid.proj.to_enu(events[i].lat_deg, events[i].lon_deg);
    ev_w[i] = std::pow(10.0, 0.4 * std::max(0.0, events[i].magnitude - 2.0));
  }

  for (std::size_t r = 0; r < grid.n_lat; ++r) {
    for (std::size_t c = 0; c < grid.n_lon; ++c) {
      const auto g_pt = grid.enu_at(r, c);
      double sum = 0.0;
      for (std::size_t i = 0; i < events.size(); ++i) {
        const double dx = g_pt.x - ev_enu[i].x;
        const double dy = g_pt.y - ev_enu[i].y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < 9.0 * kSigmaKm * kSigmaKm)
          sum += ev_w[i] * std::exp(-d2 * kInv2Sig2);
      }
      out.grid_density[r * grid.n_lon + c] = std::log1p(sum);
      max_d = std::max(max_d, out.grid_density[r * grid.n_lon + c]);
    }
  }
  if (max_d > 0.0) {
    for (double &v : out.grid_density)
      v /= max_d;
  }
  return out;
}

} // namespace chaosmap::seis
