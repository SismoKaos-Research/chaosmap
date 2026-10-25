#pragma once

// Hypocentre plane fitting (horizontal weighted PCA for strike and length plus
// a strike-normal cross-section regression for dip) for extracting pre-event
// background fault planes and post-event aftershock rupture planes.

#include "core/linalg.hpp"
#include "geo/projection.hpp"
#include "io/catalog.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <span>
#include <vector>

namespace chaosmap::seis {

struct FittedPlane {
  double center_lat{};
  double center_lon{};
  double center_depth_km{};
  double strike_deg{}; // 0..360, dipping to the right of strike if resolved
  double dip_deg{};    // 0..90, NaN if the catalog depths cannot resolve it
  double length_km{};
  double width_km{};     // NaN if the dip is unresolved
  double thickness_km{}; // 1-sigma fault-normal thickness, NaN if unresolved
  double planarity{};    // horizontal elongation 1 - lambda_0 / lambda_1
  double trace_start_lat{};
  double trace_start_lon{};
  double trace_end_lat{};
  double trace_end_lon{};
  std::size_t n_events{0};
  // Events whose depth is not held at a fixed value (used for the dip)
  std::size_t n_free_depth{0};
};

struct SeismicityAnalysis {
  FittedPlane dominant_plane;
  std::vector<FittedPlane> segments; // equal-count along-strike sub-planes
  std::vector<double> grid_density;  // normalized [0, 1] on Grid2D
};

namespace detail {

// Flags hypocentres whose depth sits in a spike of the depth histogram.
// Catalog locators hold poorly constrained events at, or barely move them
// from, a fixed trial depth (e.g. 5 or 7 km); those depths carry no
// information on dip and stack into a flat layer that pulls a fitted plane
// towards horizontal. A 0.1 km bin is a spike when it holds at least 2% of
// the events and at least 4x the median count of the bins 0.3-1.0 km away on
// either side; real depth scatter spreads over many bins and never qualifies.
[[nodiscard]] inline std::vector<std::uint8_t>
fixed_depth_mask(std::span<const io::CatalogEvent> events) {
  std::vector<std::uint8_t> fixed(events.size(), 0);
  if (events.size() < 50)
    return fixed;
  auto bin_of = [](double depth_km) { return std::lround(depth_km * 10.0); };
  std::map<long, std::size_t> counts;
  for (const auto &e : events)
    ++counts[bin_of(e.depth_km)];

  const double n = static_cast<double>(events.size());
  std::set<long> spikes;
  for (const auto &[b, c] : counts) {
    if (static_cast<double>(c) < 0.02 * n)
      continue;
    std::vector<double> nb;
    for (long k = 3; k <= 10; ++k) {
      for (long off : {-k, k}) {
        const auto it = counts.find(b + off);
        nb.push_back(it == counts.end() ? 0.0
                                        : static_cast<double>(it->second));
      }
    }
    if (static_cast<double>(c) >= 4.0 * std::max(1.0, linalg::median(nb)))
      spikes.insert(b);
  }
  for (std::size_t i = 0; i < events.size(); ++i)
    fixed[i] = spikes.contains(bin_of(events[i].depth_km)) ? 1 : 0;
  return fixed;
}

} // namespace detail

// Minimum number of free-depth events in the cross-section to fit a dip.
inline constexpr std::size_t kMinDipEvents = 20;
// The cross-section cloud must be elongated: minor/major variance <= this.
inline constexpr double kMaxDipAnisotropy = 0.25;
// A near-horizontal major axis is what a depth-limited seismogenic layer
// produces whatever the fault dip, so shallower fits are unresolved.
inline constexpr double kMinResolvedDipDeg = 10.0;

// Fits a fault plane to a cloud of earthquake hypocentres in two stages
// (weighted by M^2 so larger shocks anchor the plane): a horizontal 2D PCA
// gives strike and length, and an orthogonal regression in the strike-normal
// cross-section gives dip and width. The dip uses only events whose depths
// are not held at a fixed value (`detail::fixed_depth_mask`). If those cannot
// define a dipping plane, dip, width and thickness are NaN and the strike is
// left as the horizontal axis (its dip direction is unknown).
[[nodiscard]] inline FittedPlane
fit_hypocentre_plane(std::span<const io::CatalogEvent> events,
                     const geo::LocalProjection &proj) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  FittedPlane out;
  out.n_events = events.size();
  if (events.size() < 4) {
    out.dip_deg = kNaN;
    out.width_km = kNaN;
    out.thickness_km = kNaN;
    return out;
  }

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

  // Stage 1: 2D horizontal weighted PCA (East-North) for fault strike and
  // length
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
  // eig_xy.values[1] >= eig_xy.values[0]; vectors[1] is along-strike unit
  // vector
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

  const auto fixed = detail::fixed_depth_mask(events);
  std::vector<std::array<double, 3>> cs; // (u, v, weight) in the corridor
  double mu = 0.0, mv = 0.0, w_cs = 0.0;
  for (std::size_t i = 0; i < events.size(); ++i) {
    if (fixed[i])
      continue;
    ++out.n_free_depth;
    const double dx = pts[i].x - mean.x;
    const double dy = pts[i].y - mean.y;
    const double u = dx * perp_x + dy * perp_y; // across-strike (+ right, km)
    const double v = -pts[i].z;                 // depth (+ downward, km)
    if (std::abs(u) > 1.25 * sigma_perp)
      continue;
    cs.push_back({u, v, w[i]});
    mu += w[i] * u;
    mv += w[i] * v;
    w_cs += w[i];
  }
  // Orthogonal regression about the centroid of the cross-section subset
  double cuu = 0.0, cuv = 0.0, cvv = 0.0;
  if (w_cs > 0.0) {
    mu /= w_cs;
    mv /= w_cs;
    for (const auto &[u, v, wi] : cs) {
      cuu += wi * (u - mu) * (u - mu);
      cuv += wi * (u - mu) * (v - mv);
      cvv += wi * (v - mv) * (v - mv);
    }
    cuu /= w_cs;
    cuv /= w_cs;
    cvv /= w_cs;
  }

  const auto eig_uv = linalg::sym_eig2(cuu, cuv, cvv);
  const double du = std::abs(eig_uv.vectors[1].x);
  const double dv = std::abs(eig_uv.vectors[1].y);
  const double dip = geo::rad2deg(std::atan2(dv, std::max(1e-6, du)));
  const bool resolved =
      cs.size() >= kMinDipEvents && eig_uv.values[1] > 0.0 &&
      eig_uv.values[0] <= kMaxDipAnisotropy * eig_uv.values[1] &&
      dip >= kMinResolvedDipDeg;

  // If cuv < 0, depth increases to the left of strike_deg, so flip strike by
  // 180 deg to obey the right-hand rule (dip to the right of strike).
  if (resolved && cuv < 0.0)
    strike_deg = std::fmod(strike_deg + 180.0, 360.0);
  out.strike_deg = strike_deg;

  out.dip_deg = resolved ? dip : kNaN;
  out.thickness_km =
      resolved ? std::sqrt(std::max(0.0, eig_uv.values[0])) : kNaN;
  out.width_km =
      resolved ? 3.2 * std::sqrt(std::max(0.0, eig_uv.values[1])) : kNaN;
  out.length_km = 3.2 * std::sqrt(std::max(0.0, eig_xy.values[1]));
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

// Performs regional seismicity analysis (dominant plane, along-strike
// sub-planes, and gridded density/planarity on `grid`).
[[nodiscard]] inline SeismicityAnalysis
analyze_seismicity(std::span<const io::CatalogEvent> events,
                   const geo::Grid2D &grid, std::size_t n_segments = 3) {
  SeismicityAnalysis out;
  out.dominant_plane = fit_hypocentre_plane(events, grid.proj);
  out.grid_density.assign(grid.size(), 0.0);
  if (events.empty())
    return out;

  // Partition events along the dominant strike axis into `n_segments`
  // equal-count sub-segments to capture fault bends/splays.
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
