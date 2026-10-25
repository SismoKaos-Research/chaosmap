#pragma once

// Case-study configuration loaded from a TOML file (`configs/*.toml`).

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml++/toml.hpp>
#include <vector>

namespace chaosmap::core {

struct EventConfig {
  std::string id;
  std::string name;
  std::string date_iso;      // YYYY-MM-DD
  std::string origin_utc;    // YYYY-MM-DDTHH:MM:SS
  double mjd{0.0};           // Modified Julian Date of event day
  int64_t origin_epoch_s{0}; // Unix epoch seconds
  double lat{0.0};
  double lon{0.0};
  double depth_km{10.0};
  double mw_ref{7.0};
  double strike_hint_deg{0.0};
  double dip_hint_deg{45.0};
  double rake_hint_deg{90.0};
  bool is_strike_slip{false};
};

struct RegionConfig {
  double lat_min{37.0};
  double lat_max{40.0};
  double lon_min{41.5};
  double lon_max{45.0};
  double grid_step_deg{0.05};
  double max_gnss_dist_km{220.0};
  double max_seis_dist_km{120.0};
};

struct PathsConfig {
  std::filesystem::path gnss_dir{"data/gnss"};
  std::filesystem::path catalog_csv{
      "/home/oguzb/Projects/sismokaos/chaos/catalog_current.csv"};
  std::filesystem::path station_coords_csv{
      "/home/oguzb/Projects/sismokaos/chaos/station_coords.csv"};
  std::filesystem::path broadband_dir;
  std::vector<std::string> broadband_stations;
  std::filesystem::path out_dir{"out"};
};

struct CaseConfig {
  EventConfig event;
  RegionConfig region;
  PathsConfig paths;

  static CaseConfig load(const std::filesystem::path &path) {
    auto tbl = toml::parse_file(path.string());
    CaseConfig cfg;

    auto ev = tbl["event"];
    cfg.event.id = ev["id"].value_or(std::string{"case"});
    cfg.event.name = ev["name"].value_or(cfg.event.id);
    cfg.event.date_iso = ev["date"].value_or(std::string{"2011-10-23"});
    cfg.event.origin_utc =
        ev["origin_utc"].value_or(cfg.event.date_iso + "T00:00:00");
    cfg.event.mjd = ev["mjd"].value_or(0.0);
    cfg.event.origin_epoch_s = ev["origin_epoch_s"].value_or(int64_t{0});
    cfg.event.lat = ev["lat"].value_or(0.0);
    cfg.event.lon = ev["lon"].value_or(0.0);
    cfg.event.depth_km = ev["depth_km"].value_or(12.0);
    cfg.event.mw_ref = ev["mw_ref"].value_or(7.0);
    cfg.event.strike_hint_deg = ev["strike_hint_deg"].value_or(250.0);
    cfg.event.dip_hint_deg = ev["dip_hint_deg"].value_or(50.0);
    cfg.event.rake_hint_deg = ev["rake_hint_deg"].value_or(90.0);
    cfg.event.is_strike_slip = ev["is_strike_slip"].value_or(false);

    auto reg = tbl["region"];
    cfg.region.lat_min = reg["lat_min"].value_or(cfg.event.lat - 1.5);
    cfg.region.lat_max = reg["lat_max"].value_or(cfg.event.lat + 1.5);
    cfg.region.lon_min = reg["lon_min"].value_or(cfg.event.lon - 2.0);
    cfg.region.lon_max = reg["lon_max"].value_or(cfg.event.lon + 2.0);
    cfg.region.grid_step_deg = reg["grid_step_deg"].value_or(0.05);
    cfg.region.max_gnss_dist_km = reg["max_gnss_dist_km"].value_or(220.0);
    cfg.region.max_seis_dist_km = reg["max_seis_dist_km"].value_or(120.0);

    auto p = tbl["paths"];
    if (auto v = p["gnss_dir"].value<std::string>())
      cfg.paths.gnss_dir = *v;
    if (auto v = p["catalog_csv"].value<std::string>())
      cfg.paths.catalog_csv = *v;
    if (auto v = p["station_coords_csv"].value<std::string>())
      cfg.paths.station_coords_csv = *v;
    if (auto v = p["broadband_dir"].value<std::string>())
      cfg.paths.broadband_dir = *v;
    if (auto v = p["out_dir"].value<std::string>())
      cfg.paths.out_dir = *v;
    if (auto arr = p["broadband_stations"].as_array()) {
      for (const auto &el : *arr) {
        if (auto s = el.value<std::string>())
          cfg.paths.broadband_stations.push_back(*s);
      }
    }

    // Resolve relative paths against the repository root (parent of `configs/`)
    const auto cfg_abs = std::filesystem::absolute(path);
    const auto repo_root = (cfg_abs.parent_path().filename() == "configs")
                               ? cfg_abs.parent_path().parent_path()
                               : cfg_abs.parent_path();
    if (cfg.paths.gnss_dir.is_relative())
      cfg.paths.gnss_dir = repo_root / cfg.paths.gnss_dir;
    if (cfg.paths.out_dir.is_relative())
      cfg.paths.out_dir = repo_root / cfg.paths.out_dir;
    return cfg;
  }
};

} // namespace chaosmap::core
