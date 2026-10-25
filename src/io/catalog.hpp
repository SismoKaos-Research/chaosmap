#pragma once

// Fast AFAD earthquake catalog (`catalog_current.csv`) and seismic station
// coordinates (`station_coords.csv`) parsers.

#include "geo/projection.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace chaosmap::io {

struct CatalogEvent {
  int64_t epoch_s{}; // UTC Unix timestamp (seconds)
  double lat_deg{};
  double lon_deg{};
  double depth_km{};
  double magnitude{};
};

struct SeismicStation {
  std::string network;
  std::string station;
  double lat_deg{};
  double lon_deg{};
  double elevation_m{};
};

// Parses `DD/MM/YYYY HH:MM:SS` (or `YYYY-MM-DD HH:MM:SS`) into UTC epoch seconds.
[[nodiscard]] inline bool parse_catalog_time(std::string_view s,
                                             int64_t &out_epoch_s) noexcept {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
    s.remove_prefix(1);
  if (s.size() < 19)
    return false;

  int a = 0, b = 0, c = 0, hh = 0, mm = 0, ss = 0;
  if (s[2] == '/' && s[5] == '/') {
    // DD/MM/YYYY HH:MM:SS
    std::from_chars(s.data() + 0, s.data() + 2, a);
    std::from_chars(s.data() + 3, s.data() + 5, b);
    std::from_chars(s.data() + 6, s.data() + 10, c);
    std::from_chars(s.data() + 11, s.data() + 13, hh);
    std::from_chars(s.data() + 14, s.data() + 16, mm);
    std::from_chars(s.data() + 17, s.data() + 19, ss);
    using namespace std::chrono;
    const sys_days day =
        year_month_day{std::chrono::year{c},
                       std::chrono::month{static_cast<unsigned>(b)},
                       std::chrono::day{static_cast<unsigned>(a)}};
    out_epoch_s = day.time_since_epoch().count() * 86400LL + hh * 3600LL +
                  mm * 60LL + ss;
    return true;
  }
  if (s[4] == '-' && s[7] == '-') {
    // YYYY-MM-DD HH:MM:SS
    std::from_chars(s.data() + 0, s.data() + 4, c);
    std::from_chars(s.data() + 5, s.data() + 7, b);
    std::from_chars(s.data() + 8, s.data() + 10, a);
    std::from_chars(s.data() + 11, s.data() + 13, hh);
    std::from_chars(s.data() + 14, s.data() + 16, mm);
    std::from_chars(s.data() + 17, s.data() + 19, ss);
    using namespace std::chrono;
    const sys_days day =
        year_month_day{std::chrono::year{c},
                       std::chrono::month{static_cast<unsigned>(b)},
                       std::chrono::day{static_cast<unsigned>(a)}};
    out_epoch_s = day.time_since_epoch().count() * 86400LL + hh * 3600LL +
                  mm * 60LL + ss;
    return true;
  }
  return false;
}

// Reads AFAD catalog CSV, optionally filtering by bounding box and magnitude
// on the fly so scanning 576k rows takes ~50 ms.
[[nodiscard]] inline std::vector<CatalogEvent>
read_catalog_csv(const std::filesystem::path &path, double lat_min = -90.0,
                 double lat_max = 90.0, double lon_min = -180.0,
                 double lon_max = 180.0, double min_mag = 0.0) {
  std::vector<CatalogEvent> out;
  std::ifstream in(path);
  if (!in)
    return out;

  std::string line;
  if (!std::getline(in, line))
    return out; // skip header (and any UTF-8 BOM)

  while (std::getline(in, line)) {
    if (line.size() < 25)
      continue;
    std::string_view sv(line);
    const auto c1 = sv.find(',');
    if (c1 == std::string_view::npos)
      continue;
    int64_t t_sec = 0;
    if (!parse_catalog_time(sv.substr(0, c1), t_sec))
      continue;

    const auto c2 = sv.find(',', c1 + 1);
    const auto c3 = (c2 != std::string_view::npos) ? sv.find(',', c2 + 1)
                                                   : std::string_view::npos;
    const auto c4 = (c3 != std::string_view::npos) ? sv.find(',', c3 + 1)
                                                   : std::string_view::npos;
    const auto c5 = (c4 != std::string_view::npos) ? sv.find(',', c4 + 1)
                                                   : std::string_view::npos;
    const auto c6 = (c5 != std::string_view::npos) ? sv.find(',', c5 + 1)
                                                   : std::string_view::npos;
    const auto c7 = (c6 != std::string_view::npos) ? sv.find(',', c6 + 1)
                                                   : std::string_view::npos;
    if (c6 == std::string_view::npos)
      continue;

    double lon = 0.0, lat = 0.0, depth = 0.0, mag = 0.0;
    std::from_chars(sv.data() + c1 + 1, sv.data() + c2, lon);
    std::from_chars(sv.data() + c2 + 1, sv.data() + c3, lat);
    if (lat < lat_min || lat > lat_max || lon < lon_min || lon > lon_max)
      continue;
    std::from_chars(sv.data() + c3 + 1, sv.data() + c4, depth);
    const auto mag_end = (c7 != std::string_view::npos) ? c7 : sv.size();
    std::from_chars(sv.data() + c6 + 1, sv.data() + mag_end, mag);
    if (mag < min_mag)
      continue;

    out.push_back(CatalogEvent{
        .epoch_s = t_sec,
        .lat_deg = lat,
        .lon_deg = lon,
        .depth_km = depth,
        .magnitude = mag,
    });
  }
  return out;
}

[[nodiscard]] inline std::vector<SeismicStation>
read_station_coords_csv(const std::filesystem::path &path) {
  std::vector<SeismicStation> out;
  auto strip_quotes = [](std::string s) {
    std::erase_if(s, [](char c) { return c == '"' || c == '\r' || c == ' '; });
    return s;
  };
  auto parse_file = [&](const std::filesystem::path &p, bool lon_before_lat) {
    std::ifstream in(p);
    if (!in)
      return;
    std::string line;
    if (!std::getline(in, line))
      return;
    if (line.find("Longitude,Latitude") != std::string::npos)
      lon_before_lat = true;
    while (std::getline(in, line)) {
      if (line.empty())
        continue;
      std::stringstream ss(line);
      std::string net, sta, c1_s, c2_s, elev_s;
      if (!std::getline(ss, net, ',') || !std::getline(ss, sta, ',') ||
          !std::getline(ss, c1_s, ',') || !std::getline(ss, c2_s, ','))
        continue;
      std::getline(ss, elev_s, ',');
      SeismicStation s;
      s.network = strip_quotes(net);
      s.station = strip_quotes(sta);
      try {
        const double v1 = std::stod(strip_quotes(c1_s));
        const double v2 = std::stod(strip_quotes(c2_s));
        s.lat_deg = lon_before_lat ? v2 : v1;
        s.lon_deg = lon_before_lat ? v1 : v2;
        if (!elev_s.empty())
          s.elevation_m = std::stod(strip_quotes(elev_s));
        out.push_back(std::move(s));
      } catch (...) {
      }
    }
  };
  parse_file(path, false);
  const std::filesystem::path afad_cat =
      "/home/oguzb/Projects/sismokaos/data_downloader/catalogs/"
      "istasyon_katalog.csv";
  std::error_code ec;
  if (path != afad_cat && std::filesystem::exists(afad_cat, ec)) {
    parse_file(afad_cat, true);
  }
  return out;
}

} // namespace chaosmap::io
