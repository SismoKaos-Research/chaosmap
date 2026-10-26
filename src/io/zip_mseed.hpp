#pragma once

// Streams miniSEED-2 records directly from TDVMS `.zip` archives (via libzip)
// or loose `.mseed` files for a given station and [min_mjd, max_mjd] window.

#include "io/mseed.hpp"
#include "io/tenv3.hpp"

#include <zip.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace chaosmap::io {

namespace detail {

[[nodiscard]] inline double ymd_to_mjd_val(int y, int m, int d) noexcept {
  using namespace std::chrono;
  const sys_days sd = year_month_day{year{y}, month{static_cast<unsigned>(m)},
                                     day{static_cast<unsigned>(d)}};
  return static_cast<double>(sd.time_since_epoch().count() + 40587);
}

// Fast filter on TDVMS archive filenames of the form:
// 1) `STA_YYYY-MM-DD.zip` (typically 21-day TDVMS blocks)
// 2) `YYYYMMDDHHMM_YYYYMMDDHHMM_*.zip`
[[nodiscard]] inline bool
zip_filename_overlaps(std::string_view fname, double min_mjd, double max_mjd) {
  // Pattern 1: look for `YYYY-MM-DD` anywhere in `fname`
  for (std::size_t i = 0; i + 10 <= fname.size(); ++i) {
    if (fname[i + 4] == '-' && fname[i + 7] == '-' &&
        std::isdigit(static_cast<unsigned char>(fname[i])) &&
        std::isdigit(static_cast<unsigned char>(fname[i + 1])) &&
        std::isdigit(static_cast<unsigned char>(fname[i + 2])) &&
        std::isdigit(static_cast<unsigned char>(fname[i + 3])) &&
        std::isdigit(static_cast<unsigned char>(fname[i + 5])) &&
        std::isdigit(static_cast<unsigned char>(fname[i + 6])) &&
        std::isdigit(static_cast<unsigned char>(fname[i + 8])) &&
        std::isdigit(static_cast<unsigned char>(fname[i + 9]))) {
      const int y = std::stoi(std::string(fname.substr(i, 4)));
      const int m = std::stoi(std::string(fname.substr(i + 5, 2)));
      const int d = std::stoi(std::string(fname.substr(i + 8, 2)));
      const double z_start = ymd_to_mjd_val(y, m, d) - 0.1;
      const double z_end = z_start + 21.3;
      return !(z_end < min_mjd || z_start > max_mjd);
    }
  }

  // Pattern 2: `YYYYMMDDHHMM_YYYYMMDDHHMM_*.zip`
  if (fname.size() >= 25 && fname[12] == '_') {
    bool all_digits = true;
    for (std::size_t i = 0; i < 25; ++i) {
      if (i == 12)
        continue;
      if (!std::isdigit(static_cast<unsigned char>(fname[i]))) {
        all_digits = false;
        break;
      }
    }
    if (all_digits) {
      auto parse_ymd8 = [](std::string_view s) -> double {
        const int y = std::stoi(std::string(s.substr(0, 4)));
        const int m = std::stoi(std::string(s.substr(4, 2)));
        const int d = std::stoi(std::string(s.substr(6, 2)));
        return ymd_to_mjd_val(y, m, d);
      };
      const double z_start = parse_ymd8(fname.substr(0, 8)) - 1.0;
      const double z_end = parse_ymd8(fname.substr(13, 8)) + 1.0;
      return !(z_end < min_mjd || z_start > max_mjd);
    }
  }
  return true;
}

} // namespace detail

// Selects one seismometer component across stations whatever their sampling
// band: a SEED channel code (e.g. `BHZ`) is accepted when its first two
// letters are one of `bands` and its last letter is `component`. When a
// station records several accepted bands, the earliest listed one is kept.
struct ChannelSelector {
  std::vector<std::string> bands{"BH", "HH"};
  char component{'Z'};

  // Position of `channel`'s band in `bands`, or -1 if it is not accepted.
  [[nodiscard]] int band_rank(std::string_view channel) const noexcept {
    if (channel.size() != 3 || channel[2] != component)
      return -1;
    for (std::size_t i = 0; i < bands.size(); ++i) {
      if (channel.substr(0, 2) == bands[i])
        return static_cast<int>(i);
    }
    return -1;
  }
};

// Reads all miniSEED records of the channel picked by `select` from
// `station_dir` (containing `.zip` and/or `.mseed` files) within
// `[min_mjd, max_mjd]` and assembles them into continuous traces.
[[nodiscard]] inline std::vector<mseed::ContinuousTrace>
load_station_waveforms(const std::filesystem::path &station_dir, double min_mjd,
                       double max_mjd, const ChannelSelector &select = {}) {
  std::vector<mseed::DecodedRecord> records;
  std::error_code ec;
  if (!std::filesystem::exists(station_dir, ec))
    return {};

  std::vector<std::filesystem::path> paths;
  for (const auto &entry :
       std::filesystem::directory_iterator(station_dir, ec)) {
    if (entry.is_regular_file())
      paths.push_back(entry.path());
  }
  std::ranges::sort(paths);

  constexpr std::size_t kChunkBytes = 256 * 4096;
  std::vector<std::uint8_t> chunk(kChunkBytes);

  bool reached_max_mjd = false;

  auto scan_buffer_prefix =
      [&](std::span<const std::uint8_t> buf) -> std::size_t {
    std::size_t pos = 0;
    while (pos + mseed::kRecordBytes <= buf.size()) {
      const auto hdr_span = buf.subspan(pos, mseed::kRecordBytes);
      mseed::RecordHeader hdr;
      if (!mseed::parse_record_header(hdr_span, hdr)) {
        pos += mseed::kRecordBytes;
        continue;
      }
      const std::size_t rec_len =
          (hdr.rec_len_exp >= 8 && hdr.rec_len_exp <= 16)
              ? (std::size_t{1} << hdr.rec_len_exp)
              : mseed::kRecordBytes;
      if (pos + rec_len > buf.size())
        break; // need more bytes in next chunk

      const auto rec_span = buf.subspan(pos, rec_len);
      pos += rec_len;

      if (hdr.sample_rate_hz <= 0.0 || hdr.num_samples == 0)
        continue;
      if (select.band_rank(hdr.channel) < 0)
        continue;

      const double dur_days =
          (static_cast<double>(hdr.num_samples) / hdr.sample_rate_hz) / 86400.0;
      const double end_mjd = hdr.start_mjd + dur_days;
      if (hdr.start_mjd > max_mjd) {
        reached_max_mjd = true;
        break;
      }
      if (end_mjd < min_mjd)
        continue;
      mseed::DecodedRecord rec;
      if (mseed::decode_record(rec_span, rec)) {
        records.push_back(std::move(rec));
      }
    }
    return pos;
  };

  for (const auto &p : paths) {
    const auto ext = p.extension().string();
    if (ext == ".zip") {
      if (!detail::zip_filename_overlaps(p.filename().string(), min_mjd,
                                         max_mjd))
        continue;

      int err = 0;
      zip_t *za = zip_open(p.c_str(), ZIP_RDONLY, &err);
      if (!za)
        continue;

      const zip_int64_t n_entries = zip_get_num_entries(za, 0);
      for (zip_int64_t idx = 0; idx < n_entries; ++idx) {
        zip_stat_t st;
        if (zip_stat_index(za, static_cast<zip_uint64_t>(idx), 0, &st) != 0)
          continue;
        if ((st.valid & ZIP_STAT_SIZE) == 0 || st.size < mseed::kRecordBytes)
          continue;

        zip_file_t *zf = zip_fopen_index(za, static_cast<zip_uint64_t>(idx), 0);
        if (!zf)
          continue;

        std::size_t carry = 0;
        reached_max_mjd = false;
        while (!reached_max_mjd) {
          const zip_int64_t nr =
              zip_fread(zf, chunk.data() + carry, kChunkBytes - carry);
          if (nr <= 0)
            break;
          const std::size_t total = carry + static_cast<std::size_t>(nr);
          const std::size_t consumed = scan_buffer_prefix(
              std::span<const std::uint8_t>(chunk.data(), total));
          if (reached_max_mjd)
            break;
          if (consumed == 0 && total == kChunkBytes) {
            carry = 0;
            continue;
          }
          carry = total - consumed;
          if (carry > 0 && consumed > 0) {
            std::memmove(chunk.data(), chunk.data() + consumed, carry);
          }
        }
        zip_fclose(zf);
      }
      zip_close(za);
    } else if (ext == ".mseed" || ext == ".msd") {
      std::ifstream in(p, std::ios::binary);
      if (!in)
        continue;
      std::size_t carry = 0;
      reached_max_mjd = false;
      while (in) {
        in.read(reinterpret_cast<char *>(chunk.data() + carry),
                static_cast<std::streamsize>(kChunkBytes - carry));
        const auto got = static_cast<std::size_t>(in.gcount());
        if (got == 0)
          break;
        const std::size_t total = carry + got;
        const std::size_t consumed = scan_buffer_prefix(
            std::span<const std::uint8_t>(chunk.data(), total));
        carry = total - consumed;
        if (carry > 0 && consumed > 0) {
          std::memmove(chunk.data(), chunk.data() + consumed, carry);
        }
      }
    }
  }

  auto traces = mseed::assemble_traces(records);
  int best_rank = std::numeric_limits<int>::max();
  for (const auto &tr : traces)
    best_rank = std::min(best_rank, select.band_rank(tr.channel));
  std::erase_if(traces, [&](const mseed::ContinuousTrace &tr) {
    return select.band_rank(tr.channel) != best_rank;
  });
  return traces;
}

} // namespace chaosmap::io
