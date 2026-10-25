#pragma once

// Exports all chaosmap products for a case study into `out/<case_id>/`:
// - `summary_metrics.json`
// - `stations_gnss.csv`
// - `slip_patches.csv`
// - `lyapunov_stations.csv` & `lyapunov_daily.csv`
// - `grids.csv`
// - `fault_features.geojson` (RFC 7946 GeoJSON for GIS / web viewers).

#include "anomaly/array_anomaly.hpp"
#include "core/config.hpp"
#include "fusion/benchmark.hpp"
#include "fusion/fuse.hpp"
#include "geo/projection.hpp"
#include "gnss/fields.hpp"
#include "gnss/trajectory.hpp"
#include "okada/inversion.hpp"
#include "seis/planes.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <span>
#include <string>

namespace chaosmap::out {

namespace detail {

[[nodiscard]] inline double safe_num(double v) noexcept {
  return std::isfinite(v) ? v : 0.0;
}

} // namespace detail

inline void export_case_results(
    const core::CaseConfig &cfg, const geo::Grid2D &grid,
    std::span<const gnss::StationMotion> gnss_stations,
    const gnss::GeodeticGridFields &geod, const okada::InversionResult &inv,
    const seis::SeismicityAnalysis &pre_seis,
    const seis::SeismicityAnalysis &post_seis,
    const anomaly::ArrayAnomalyResult &seis_anom,
    const fusion::LyapunovBenchmark &bm, const fusion::FusedFaultMap &fused) {
  namespace fs = std::filesystem;
  const fs::path out_dir = cfg.paths.out_dir;
  fs::create_directories(out_dir);

  // 1. summary_metrics.json
  {
    std::ofstream f(out_dir / "summary_metrics.json");
    f << std::fixed << std::setprecision(5);
    f << "{\n";
    f << "  \"case_id\": \"" << cfg.event.id << "\",\n";
    f << "  \"event_name\": \"" << cfg.event.name << "\",\n";
    f << "  \"date_iso\": \"" << cfg.event.date_iso << "\",\n";
    f << "  \"event_mjd\": " << detail::safe_num(cfg.event.mjd) << ",\n";
    f << "  \"hypocentre\": {\"lat\": " << cfg.event.lat
      << ", \"lon\": " << cfg.event.lon
      << ", \"depth_km\": " << cfg.event.depth_km
      << ", \"catalog_mw\": " << cfg.event.mw_ref << "},\n";

    f << "  \"how_much_it_moved\": {\n";
    f << "    \"max_disp_station\": \"" << fused.max_disp_station << "\",\n";
    f << "    \"max_station_horiz_m\": "
      << detail::safe_num(fused.max_station_horiz_m) << ",\n";
    f << "    \"max_station_horiz_mm\": "
      << detail::safe_num(fused.max_station_horiz_m * 1000.0) << ",\n";
    f << "    \"max_station_up_m\": "
      << detail::safe_num(fused.max_station_up_m) << ",\n";
    f << "    \"peak_fault_slip_m\": " << detail::safe_num(fused.peak_slip_m)
      << ",\n";
    f << "    \"top_quartile_fault_slip_m\": "
      << detail::safe_num(fused.top_quartile_slip_m) << ",\n";
    f << "    \"mean_fault_slip_m\": " << detail::safe_num(fused.mean_slip_m)
      << ",\n";
    f << "    \"geodetic_moment_nm\": " << std::scientific
      << detail::safe_num(inv.moment_nm) << std::fixed << ",\n";
    f << "    \"geodetic_mw\": " << detail::safe_num(fused.geodetic_mw)
      << ",\n";
    f << "    \"variance_reduction_pct\": "
      << detail::safe_num(fused.variance_reduction_pct) << "\n";
    f << "  },\n";

    f << "  \"coseismic_fault_geometry\": {\n";
    f << "    \"center_lat\": " << detail::safe_num(inv.center_lat_deg)
      << ",\n";
    f << "    \"center_lon\": " << detail::safe_num(inv.center_lon_deg)
      << ",\n";
    f << "    \"center_depth_km\": " << detail::safe_num(inv.plane.depth_km)
      << ",\n";
    f << "    \"length_km\": " << detail::safe_num(inv.plane.length_km)
      << ",\n";
    f << "    \"width_km\": " << detail::safe_num(inv.plane.width_km) << ",\n";
    f << "    \"strike_deg\": " << detail::safe_num(inv.plane.strike_deg)
      << ",\n";
    f << "    \"dip_deg\": " << detail::safe_num(inv.plane.dip_deg) << ",\n";
    f << "    \"mean_rake_deg\": " << detail::safe_num(inv.mean_rake_deg)
      << ",\n";
    f << "    \"mean_slip_m\": " << detail::safe_num(inv.mean_slip_m) << "\n";
    f << "  },\n";

    f << "  \"before_vs_after_fault_comparison\": {\n";
    f << "    \"before_strike_deg\": "
      << detail::safe_num(fused.before_strike_deg) << ",\n";
    f << "    \"coseismic_strike_deg\": "
      << detail::safe_num(fused.coseismic_strike_deg) << ",\n";
    f << "    \"after_strike_deg\": "
      << detail::safe_num(fused.after_strike_deg) << ",\n";
    f << "    \"strike_change_deg\": "
      << detail::safe_num(fused.strike_change_deg) << ",\n";
    f << "    \"centroid_shift_km\": "
      << detail::safe_num(fused.centroid_shift_km) << ",\n";
    f << "    \"pre_events_count\": " << pre_seis.dominant_plane.n_events
      << ",\n";
    f << "    \"post_aftershocks_count\": " << post_seis.dominant_plane.n_events
      << ",\n";
    f << "    \"post_aftershock_length_km\": "
      << detail::safe_num(post_seis.dominant_plane.length_km) << ",\n";
    f << "    \"post_aftershock_dip_deg\": "
      << detail::safe_num(post_seis.dominant_plane.dip_deg) << "\n";
    f << "  },\n";

    f << "  \"lyapunov_benchmark\": {\n";
    f << "    \"selected_mode\": \"" << bm.best_candidate << "\",\n";
    f << "    \"l_geo\": {\n";
    f << "      \"available\": " << (bm.l_geo.available ? "true" : "false")
      << ",\n";
    f << "      \"ridge_to_rupture_dist_km\": "
      << detail::safe_num(bm.l_geo.ridge_to_rupture_dist_km) << ",\n";
    f << "      \"strike_misfit_deg\": "
      << detail::safe_num(bm.l_geo.strike_misfit_deg) << ",\n";
    f << "      \"spatial_roc_auc\": "
      << detail::safe_num(bm.l_geo.spatial_roc_auc) << ",\n";
    f << "      \"composite_skill\": "
      << detail::safe_num(bm.l_geo.composite_skill) << ",\n";
    f << "      \"assigned_weight\": "
      << detail::safe_num(bm.l_geo.assigned_weight) << "\n";
    f << "    },\n";
    f << "    \"l_seis\": {\n";
    f << "      \"available\": " << (bm.l_seis.available ? "true" : "false")
      << ",\n";
    f << "      \"ridge_to_rupture_dist_km\": "
      << detail::safe_num(bm.l_seis.ridge_to_rupture_dist_km) << ",\n";
    f << "      \"strike_misfit_deg\": "
      << detail::safe_num(bm.l_seis.strike_misfit_deg) << ",\n";
    f << "      \"spatial_roc_auc\": "
      << detail::safe_num(bm.l_seis.spatial_roc_auc) << ",\n";
    f << "      \"temporal_pre_auc\": "
      << detail::safe_num(bm.l_seis.temporal_pre_auc) << ",\n";
    f << "      \"null_p_value\": "
      << detail::safe_num(seis_anom.array_null_p_value) << ",\n";
    f << "      \"composite_skill\": "
      << detail::safe_num(bm.l_seis.composite_skill) << ",\n";
    f << "      \"assigned_weight\": "
      << detail::safe_num(bm.l_seis.assigned_weight) << "\n";
    f << "    }\n";
    f << "  }\n";
    f << "}\n";
  }

  // 2. stations_gnss.csv
  {
    std::ofstream f(out_dir / "stations_gnss.csv");
    f << std::fixed << std::setprecision(5);
    f << "station,lat,lon,dist_km,ve_mm_yr,vn_mm_yr,vu_mm_yr,co_e_mm,co_n_mm,"
         "co_u_mm,co_horiz_mm,post_e_mm,post_n_mm,has_vel,has_coseismic\n";
    for (const auto &s : gnss_stations) {
      const double dist = geo::haversine_km(s.lat_deg, s.lon_deg,
                                            cfg.event.lat, cfg.event.lon);
      const double co_h =
          std::hypot(s.coseismic_enu_m.x, s.coseismic_enu_m.y) * 1000.0;
      f << s.station << ',' << s.lat_deg << ',' << s.lon_deg << ',' << dist
        << ',' << s.vel_enu_m_yr.x * 1000.0 << ',' << s.vel_enu_m_yr.y * 1000.0
        << ',' << s.vel_enu_m_yr.z * 1000.0 << ','
        << s.coseismic_enu_m.x * 1000.0 << ',' << s.coseismic_enu_m.y * 1000.0
        << ',' << s.coseismic_enu_m.z * 1000.0 << ',' << co_h << ','
        << s.postseismic_60d_enu_m.x * 1000.0 << ','
        << s.postseismic_60d_enu_m.y * 1000.0 << ',' << (s.has_velocity ? 1 : 0)
        << ',' << (s.has_coseismic ? 1 : 0) << '\n';
    }
  }

  // 3. slip_patches.csv
  {
    std::ofstream f(out_dir / "slip_patches.csv");
    f << std::fixed << std::setprecision(5);
    f << "patch_id,lat,lon,depth_km,strike_deg,dip_deg,length_km,width_km,"
         "strike_slip_m,dip_slip_m,total_slip_m,rake_deg,resolution\n";
    for (std::size_t i = 0; i < inv.patches.size(); ++i) {
      const auto &p = inv.patches[i];
      f << i << ',' << p.lat_deg << ',' << p.lon_deg << ',' << p.patch.depth_km
        << ',' << p.patch.strike_deg << ',' << p.patch.dip_deg << ','
        << p.patch.length_km << ',' << p.patch.width_km << ','
        << p.patch.slip_strike_m << ',' << p.patch.slip_dip_m << ','
        << p.total_slip_m << ',' << p.rake_deg << ',' << p.resolution << '\n';
    }
  }

  // 4. lyapunov_stations.csv & lyapunov_daily.csv
  if (seis_anom.has_seismic) {
    std::ofstream fs_sta(out_dir / "lyapunov_stations.csv");
    fs_sta << std::fixed << std::setprecision(5);
    fs_sta << "station,lat,lon,dist_km,pre_wolf_med,post_wolf_med,delta_wolf,"
              "pre_ros_med,post_ros_med,delta_ros,pre_max_z,pre_crossband_z,"
              "station_auc\n";
    std::ofstream fs_day(out_dir / "lyapunov_daily.csv");
    fs_day << std::fixed << std::setprecision(5);
    fs_day << "station,day_mjd,rel_day,wolf_fault,wolf_ctrl,ros_short,samp_ent,"
              "corr_dim,z_iqr_wolf,z_iqr_crossband,n_windows\n";
    const int ev_day = static_cast<int>(std::floor(cfg.event.mjd));
    for (const auto &s : seis_anom.stations) {
      fs_sta << s.station << ',' << s.lat_deg << ',' << s.lon_deg << ','
             << s.dist_to_hypo_km << ',' << s.pre_wolf_median << ','
             << s.post_wolf_median << ',' << s.delta_wolf_post_minus_pre << ','
             << s.pre_ros_short_median << ',' << s.post_ros_short_median << ','
             << s.delta_ros_short << ',' << s.pre_horizon_max_z << ','
             << s.pre_horizon_crossband_z << ',' << s.station_auc << '\n';
      for (const auto &d : s.daily) {
        fs_day << s.station << ',' << d.day_mjd << ',' << (d.day_mjd - ev_day)
               << ',' << d.wolf_fault << ',' << d.wolf_ctrl << ','
               << d.ros_short_fault << ',' << d.samp_ent_fault << ','
               << d.corr_dim_fault << ',' << d.z_iqr_wolf << ','
               << d.z_iqr_crossband << ',' << d.n_windows << '\n';
      }
    }
  }

  // 5. grids.csv
  {
    std::ofstream f(out_dir / "grids.csv");
    f << std::fixed << std::setprecision(5);
    f << "row,col,lat,lon,vel_e_mm_yr,vel_n_mm_yr,max_shear_ns_yr,"
         "dilatation_ns_yr,ftle_interseismic,ftle_coseismic,pre_seis_density,"
         "post_seis_density,l_seis_pre,l_seis_post,l_seis_change,"
         "before_susceptibility,moved_surface_disp_m,moved_intensity,"
         "after_state,change_after_minus_before\n";
    for (std::size_t r = 0; r < grid.n_lat; ++r) {
      for (std::size_t c = 0; c < grid.n_lon; ++c) {
        const std::size_t idx = r * grid.n_lon + c;
        f << r << ',' << c << ',' << grid.lat_at(r) << ',' << grid.lon_at(c)
          << ',' << geod.vel_enu_m_yr[idx].x * 1000.0 << ','
          << geod.vel_enu_m_yr[idx].y * 1000.0 << ','
          << geod.max_shear_nanostrain_yr[idx] << ','
          << geod.dilatation_nanostrain_yr[idx] << ','
          << geod.ftle_interseismic[idx] << ',' << geod.ftle_coseismic[idx]
          << ',' << pre_seis.grid_density[idx] << ','
          << post_seis.grid_density[idx] << ',' << seis_anom.grid_l_seis_pre[idx]
          << ',' << seis_anom.grid_l_seis_post[idx] << ','
          << seis_anom.grid_l_seis_change[idx] << ','
          << fused.before_susceptibility[idx] << ','
          << fused.moved_surface_disp_m[idx] << ',' << fused.moved_intensity[idx]
          << ',' << fused.after_state[idx] << ','
          << fused.change_after_minus_before[idx] << '\n';
      }
    }
  }

  // 6. fault_features.geojson
  {
    std::ofstream f(out_dir / "fault_features.geojson");
    f << std::fixed << std::setprecision(5);
    f << "{\n  \"type\": \"FeatureCollection\",\n  \"features\": [\n";
    bool first = true;
    auto emit_line = [&](const std::string &role, const std::string &label,
                         double lat0, double lon0, double lat1, double lon1,
                         double strike, double dip) {
      if (!first)
        f << ",\n";
      first = false;
      f << "    {\"type\": \"Feature\", \"properties\": {\"role\": \"" << role
        << "\", \"label\": \"" << label
        << "\", \"strike_deg\": " << detail::safe_num(strike)
        << ", \"dip_deg\": " << detail::safe_num(dip)
        << "}, \"geometry\": {\"type\": \"LineString\", \"coordinates\": [["
        << lon0 << ", " << lat0 << "], [" << lon1 << ", " << lat1 << "]]}}";
    };

    const double s_rad = geo::deg2rad(inv.plane.strike_deg);
    const linalg::Vec2 s_hat{std::sin(s_rad), std::cos(s_rad)};
    const linalg::Vec2 d_hat{std::cos(s_rad), -std::sin(s_rad)};

    emit_line("coseismic_fault_trace", "Inverted Coseismic Fault Trace",
              inv.trace_start_lat, inv.trace_start_lon, inv.trace_end_lat,
              inv.trace_end_lon, inv.plane.strike_deg, inv.plane.dip_deg);

    if (pre_seis.dominant_plane.n_events >= 4) {
      emit_line("pre_seismicity_trace", "Pre-Event Seismicity Lineament",
                pre_seis.dominant_plane.trace_start_lat,
                pre_seis.dominant_plane.trace_start_lon,
                pre_seis.dominant_plane.trace_end_lat,
                pre_seis.dominant_plane.trace_end_lon,
                pre_seis.dominant_plane.strike_deg,
                pre_seis.dominant_plane.dip_deg);
    }
    if (post_seis.dominant_plane.n_events >= 4) {
      emit_line("post_seismicity_trace", "Post-Event Aftershock Plane",
                post_seis.dominant_plane.trace_start_lat,
                post_seis.dominant_plane.trace_start_lon,
                post_seis.dominant_plane.trace_end_lat,
                post_seis.dominant_plane.trace_end_lon,
                post_seis.dominant_plane.strike_deg,
                post_seis.dominant_plane.dip_deg);
    }
    for (std::size_t k = 0; k < post_seis.segments.size(); ++k) {
      const auto &seg = post_seis.segments[k];
      emit_line("aftershock_segment",
                "Aftershock Segment " + std::to_string(k + 1),
                seg.trace_start_lat, seg.trace_start_lon, seg.trace_end_lat,
                seg.trace_end_lon, seg.strike_deg, seg.dip_deg);
    }
    if (bm.l_geo.available) {
      emit_line("l_geo_ftle_ridge", "L-geo GNSS FTLE Ridge",
                geod.ridge_start_lat, geod.ridge_start_lon, geod.ridge_end_lat,
                geod.ridge_end_lon, geod.ridge_strike_deg, 90.0);
    }
    if (bm.l_seis.available) {
      emit_line("l_seis_lyapunov_ridge", "L-seis Broadband Lyapunov Ridge",
                seis_anom.ridge_start_lat, seis_anom.ridge_start_lon,
                seis_anom.ridge_end_lat, seis_anom.ridge_end_lon,
                seis_anom.ridge_strike_deg, 90.0);
    }

    // Slip patch polygons
    for (std::size_t i = 0; i < inv.patches.size(); ++i) {
      const auto &p = inv.patches[i];
      const auto pc = grid.proj.to_enu(p.lat_deg, p.lon_deg);
      const double hL = 0.5 * p.patch.length_km;
      const double hW =
          0.5 * p.patch.width_km * std::cos(geo::deg2rad(p.patch.dip_deg));
      const auto c00 =
          grid.proj.to_latlon((pc - s_hat * hL - d_hat * hW).x,
                              (pc - s_hat * hL - d_hat * hW).y);
      const auto c10 =
          grid.proj.to_latlon((pc + s_hat * hL - d_hat * hW).x,
                              (pc + s_hat * hL - d_hat * hW).y);
      const auto c11 =
          grid.proj.to_latlon((pc + s_hat * hL + d_hat * hW).x,
                              (pc + s_hat * hL + d_hat * hW).y);
      const auto c01 =
          grid.proj.to_latlon((pc - s_hat * hL + d_hat * hW).x,
                              (pc - s_hat * hL + d_hat * hW).y);
      f << ",\n    {\"type\": \"Feature\", \"properties\": {\"role\": "
           "\"slip_patch\", \"patch_id\": "
        << i << ", \"slip_m\": " << detail::safe_num(p.total_slip_m)
        << ", \"rake_deg\": " << detail::safe_num(p.rake_deg)
        << ", \"depth_km\": " << detail::safe_num(p.patch.depth_km)
        << "}, \"geometry\": {\"type\": \"Polygon\", \"coordinates\": [[["
        << c00.y << ", " << c00.x << "], [" << c10.y << ", " << c10.x << "], ["
        << c11.y << ", " << c11.x << "], [" << c01.y << ", " << c01.x << "], ["
        << c00.y << ", " << c00.x << "]]]}}";
    }

    f << "\n  ]\n}\n";
  }
}

} // namespace chaosmap::out
