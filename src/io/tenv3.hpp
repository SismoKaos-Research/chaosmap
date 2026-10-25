#pragma once

// Parser for Nevada Geodetic Laboratory (NGL) `.tenv3` daily GNSS position
// files, `stations.csv`, and `steps.txt` equipment/earthquake step database.

#include "geo/projection.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace chaosmap::io {

struct Tenv3Sample {
  double dec_year{};
  double mjd{};
  double east_m{};
  double north_m{};
  double up_m{};
  double sig_e_m{};
  double sig_n_m{};
  double sig_u_m{};
};

struct GnssStationSeries {
  std::string station;
  double lat_deg{};
  double lon_deg{};
  double height_m{};
  std::vector<Tenv3Sample> samples;
};

struct StepEntry {
  std::string station;
  double mjd{};
  int type{1}; // 1 = equipment, 2 = earthquake
  double thresh_dist_km{0.0};
  double station_dist_km{0.0};
  double magnitude{0.0};
  std::string description;
};

// Converts NGL date string `YYMMMDD` (e.g. `11OCT23`, `99AUG17`) to MJD.
[[nodiscard]] inline double parse_ngl_date_mjd(std::string_view s) noexcept {
  if (s.size() < 7)
    return 0.0;
  int yy = 0;
  int dd = 0;
  std::from_chars(s.data(), s.data() + 2, yy);
  std::from_chars(s.data() + 5, s.data() + 7, dd);
  const int year = (yy >= 80) ? (1900 + yy) : (2000 + yy);
  const std::string_view mstr = s.substr(2, 3);
  unsigned month = 1;
  constexpr std::array<std::string_view, 12> kMonths = {
      "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
      "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
  for (unsigned i = 0; i < 12; ++i) {
    if (mstr == kMonths[i]) {
      month = i + 1;
      break;
    }
  }
  using namespace std::chrono;
  const sys_days d =
      year_month_day{std::chrono::year{year}, std::chrono::month{month},
                     std::chrono::day{static_cast<unsigned>(dd)}};
  // MJD epoch is 1858-11-17 (40587 days before 1970-01-01).
  return static_cast<double>(d.time_since_epoch().count() + 40587);
}

[[nodiscard]] inline GnssStationSeries
read_tenv3(const std::filesystem::path &path) {
  GnssStationSeries out;
  out.station = path.stem().string();
  std::ifstream in(path);
  if (!in)
    return out;

  std::string line;
  // Skip header line
  if (!std::getline(in, line))
    return out;

  bool have_coords = false;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::istringstream iss(line);
    std::string site, yymmmdd;
    double dec_yr = 0.0, mjd = 0.0;
    int week = 0, dow = 0;
    double reflon = 0.0;
    double e0 = 0.0, east = 0.0, n0 = 0.0, north = 0.0, u0 = 0.0, up = 0.0;
    double ant = 0.0, sige = 0.0, sign = 0.0, sigu = 0.0;
    double cen = 0.0, ceu = 0.0, cnu = 0.0;
    double lat = 0.0, lon = 0.0, hgt = 0.0;
    if (!(iss >> site >> yymmmdd >> dec_yr >> mjd >> week >> dow >> reflon >>
          e0 >> east >> n0 >> north >> u0 >> up >> ant >> sige >> sign >>
          sigu >> cen >> ceu >> cnu >> lat >> lon >> hgt)) {
      continue;
    }
    if (!have_coords) {
      out.station = site;
      out.lat_deg = lat;
      out.lon_deg = geo::wrap_lon(lon);
      out.height_m = hgt;
      have_coords = true;
    }
    out.samples.push_back(Tenv3Sample{
        .dec_year = dec_yr,
        .mjd = mjd,
        .east_m = e0 + east,
        .north_m = n0 + north,
        .up_m = u0 + up,
        .sig_e_m = sige,
        .sig_n_m = sign,
        .sig_u_m = sigu,
    });
  }
  return out;
}

[[nodiscard]] inline std::unordered_map<std::string, std::vector<StepEntry>>
read_steps_file(const std::filesystem::path &path) {
  std::unordered_map<std::string, std::vector<StepEntry>> out;
  std::ifstream in(path);
  if (!in)
    return out;

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::istringstream iss(line);
    StepEntry e;
    std::string date_str;
    if (!(iss >> e.station >> date_str >> e.type))
      continue;
    e.mjd = parse_ngl_date_mjd(date_str);
    if (e.type == 1) {
      std::getline(iss, e.description);
    } else if (e.type == 2) {
      iss >> e.thresh_dist_km >> e.station_dist_km >> e.magnitude >>
          e.description;
    }
    out[e.station].push_back(std::move(e));
  }
  return out;
}

// Loads all `.tenv3` files inside `gnss_dir/IGS20` (or `gnss_dir/IGS14`) within
// `max_dist_km` of `(center_lat, center_lon)`.
[[nodiscard]] inline std::vector<GnssStationSeries>
load_regional_gnss(const std::filesystem::path &gnss_dir, double center_lat,
                   double center_lon, double max_dist_km) {
  std::vector<GnssStationSeries> result;
  std::filesystem::path sub = gnss_dir / "IGS20";
  if (!std::filesystem::is_directory(sub))
    sub = gnss_dir / "IGS14";
  if (!std::filesystem::is_directory(sub))
    sub = gnss_dir;
  if (!std::filesystem::is_directory(sub))
    return result;

  std::vector<std::filesystem::path> files;
  for (const auto &entry : std::filesystem::directory_iterator(sub)) {
    if (entry.is_regular_file() && entry.path().extension() == ".tenv3")
      files.push_back(entry.path());
  }
  std::ranges::sort(files);

  for (const auto &f : files) {
    auto s = read_tenv3(f);
    if (s.samples.empty())
      continue;
    if (geo::haversine_km(center_lat, center_lon, s.lat_deg, s.lon_deg) <=
        max_dist_km) {
      result.push_back(std::move(s));
    }
  }
  return result;
}

} // namespace chaosmap::io
