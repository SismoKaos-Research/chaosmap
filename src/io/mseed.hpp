#pragma once

// Zero-dependency 512-byte miniSEED-2 fixed header + Blockette 1000/1001 +
// Steim2 (encoding 11), 16-bit int (encoding 1), 32-bit int (encoding 3)
// decoder for TDVMS Turkish broadband waveforms.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace chaosmap::io::mseed {

inline constexpr std::size_t kRecordBytes = 512;

struct BTime {
  std::uint16_t year{};
  std::uint16_t doy{};
  std::uint8_t hour{};
  std::uint8_t min{};
  std::uint8_t sec{};
  std::uint8_t unused{};
  std::uint16_t frac_0001{}; // 0.0001 s ticks
};

struct RecordHeader {
  std::string station;
  std::string location;
  std::string channel;
  std::string network;
  BTime start_btime{};
  double start_epoch_s{}; // Unix epoch seconds (1970-01-01T00:00:00Z)
  double start_mjd{};     // Modified Julian Date
  std::uint16_t num_samples{};
  std::int16_t rate_factor{};
  std::int16_t rate_mult{};
  double sample_rate_hz{};
  std::uint8_t activity_flags{};
  std::uint8_t io_clock_flags{};
  std::uint8_t quality_flags{};
  std::uint8_t num_blockettes{};
  std::int32_t time_correction{};
  std::uint16_t data_offset{};
  std::uint16_t first_blockette_offset{};
  std::uint8_t encoding{11};  // 1=int16, 3=int32, 11=Steim2
  std::uint8_t byte_order{1}; // 1=big-endian
  std::uint8_t rec_len_exp{9};
  std::int8_t microsec_corr{0};
};

struct DecodedRecord {
  RecordHeader hdr;
  std::vector<double> samples;
};

struct ContinuousTrace {
  std::string network;
  std::string station;
  std::string channel;
  double sample_rate_hz{};
  double start_epoch_s{};
  double start_mjd{};
  std::vector<double> data;
};

[[nodiscard]] constexpr std::uint16_t bswap16(std::uint16_t v) noexcept {
  return std::byteswap(v);
}
[[nodiscard]] constexpr std::uint32_t bswap32(std::uint32_t v) noexcept {
  return std::byteswap(v);
}

[[nodiscard]] inline std::uint16_t rd_u16_be(const std::uint8_t *p) noexcept {
  std::uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  if constexpr (std::endian::native == std::endian::little)
    v = bswap16(v);
  return v;
}

[[nodiscard]] inline std::int16_t rd_i16_be(const std::uint8_t *p) noexcept {
  return static_cast<std::int16_t>(rd_u16_be(p));
}

[[nodiscard]] inline std::uint32_t rd_u32_be(const std::uint8_t *p) noexcept {
  std::uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  if constexpr (std::endian::native == std::endian::little)
    v = bswap32(v);
  return v;
}

[[nodiscard]] inline std::int32_t rd_i32_be(const std::uint8_t *p) noexcept {
  return static_cast<std::int32_t>(rd_u32_be(p));
}

[[nodiscard]] constexpr bool is_leap_year(int y) noexcept {
  return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

[[nodiscard]] inline double btime_to_epoch(const BTime &bt,
                                           std::int8_t usec_corr = 0) noexcept {
  int days = 0;
  for (int y = 1970; y < static_cast<int>(bt.year); ++y) {
    days += is_leap_year(y) ? 366 : 365;
  }
  days += static_cast<int>(bt.doy) - 1;
  const double sec = static_cast<double>(days) * 86400.0 +
                     static_cast<double>(bt.hour) * 3600.0 +
                     static_cast<double>(bt.min) * 60.0 +
                     static_cast<double>(bt.sec) +
                     static_cast<double>(bt.frac_0001) * 1e-4 +
                     static_cast<double>(usec_corr) * 1e-6;
  return sec;
}

[[nodiscard]] constexpr double epoch_to_mjd(double epoch_s) noexcept {
  // 1970-01-01T00:00:00Z is MJD 40587.0
  return 40587.0 + epoch_s / 86400.0;
}

[[nodiscard]] constexpr double decode_sample_rate(std::int16_t factor,
                                                  std::int16_t mult) noexcept {
  if (factor == 0 || mult == 0)
    return 0.0;
  const double f = static_cast<double>(factor);
  const double m = static_cast<double>(mult);
  if (factor > 0 && mult > 0)
    return f * m;
  if (factor > 0 && mult < 0)
    return -f / m;
  if (factor < 0 && mult > 0)
    return -m / f;
  return 1.0 / (f * m);
}

[[nodiscard]] inline std::string trim_ascii(const char *p, std::size_t n) {
  std::size_t a = 0;
  while (a < n && (p[a] == ' ' || p[a] == '\0'))
    ++a;
  std::size_t b = n;
  while (b > a && (p[b - 1] == ' ' || p[b - 1] == '\0'))
    --b;
  return std::string(p + a, b - a);
}

[[nodiscard]] inline bool parse_record_header(std::span<const std::uint8_t> rec,
                                              RecordHeader &hdr) {
  if (rec.size() < kRecordBytes)
    return false;
  const char ind = static_cast<char>(rec[6]);
  if (ind != 'D' && ind != 'R' && ind != 'Q' && ind != 'M')
    return false;

  hdr.station = trim_ascii(reinterpret_cast<const char *>(&rec[8]), 5);
  hdr.location = trim_ascii(reinterpret_cast<const char *>(&rec[13]), 2);
  hdr.channel = trim_ascii(reinterpret_cast<const char *>(&rec[15]), 3);
  hdr.network = trim_ascii(reinterpret_cast<const char *>(&rec[18]), 2);

  hdr.start_btime.year = rd_u16_be(&rec[20]);
  hdr.start_btime.doy = rd_u16_be(&rec[22]);
  hdr.start_btime.hour = rec[24];
  hdr.start_btime.min = rec[25];
  hdr.start_btime.sec = rec[26];
  hdr.start_btime.unused = rec[27];
  hdr.start_btime.frac_0001 = rd_u16_be(&rec[28]);

  hdr.num_samples = rd_u16_be(&rec[30]);
  hdr.rate_factor = rd_i16_be(&rec[32]);
  hdr.rate_mult = rd_i16_be(&rec[34]);
  hdr.sample_rate_hz = decode_sample_rate(hdr.rate_factor, hdr.rate_mult);

  hdr.activity_flags = rec[36];
  hdr.io_clock_flags = rec[37];
  hdr.quality_flags = rec[38];
  hdr.num_blockettes = rec[39];
  hdr.time_correction = rd_i32_be(&rec[40]);
  hdr.data_offset = rd_u16_be(&rec[44]);
  hdr.first_blockette_offset = rd_u16_be(&rec[46]);

  std::size_t boff = hdr.first_blockette_offset;
  for (std::uint8_t b = 0;
       b < hdr.num_blockettes && boff >= 48 && boff + 4 <= rec.size(); ++b) {
    const std::uint16_t btype = rd_u16_be(&rec[boff]);
    const std::size_t next_boff = rd_u16_be(&rec[boff + 2]);
    if (btype == 1000 && boff + 8 <= rec.size()) {
      hdr.encoding = rec[boff + 4];
      hdr.byte_order = rec[boff + 5];
      hdr.rec_len_exp = rec[boff + 6];
    } else if (btype == 1001 && boff + 8 <= rec.size()) {
      hdr.microsec_corr = static_cast<std::int8_t>(rec[boff + 5]);
    }
    if (next_boff == 0 || next_boff <= boff)
      break;
    boff = next_boff;
  }

  hdr.start_epoch_s = btime_to_epoch(hdr.start_btime, hdr.microsec_corr);
  if ((hdr.activity_flags & 0x02) == 0 && hdr.time_correction != 0) {
    hdr.start_epoch_s += static_cast<double>(hdr.time_correction) * 1e-4;
  }
  hdr.start_mjd = epoch_to_mjd(hdr.start_epoch_s);
  return true;
}

template <unsigned Bits>
[[nodiscard]] constexpr std::int32_t sign_extend(std::uint32_t val) noexcept {
  static_assert(Bits > 0 && Bits < 32);
  constexpr std::uint32_t mask = (1u << Bits) - 1u;
  val &= mask;
  constexpr std::uint32_t sign_bit = 1u << (Bits - 1u);
  return static_cast<std::int32_t>((val ^ sign_bit) - sign_bit);
}

inline void decode_steim2_payload(std::span<const std::uint8_t> payload,
                                  std::size_t num_samples,
                                  std::vector<double> &out) {
  out.clear();
  out.reserve(num_samples);
  if (num_samples == 0 || payload.size() < 64)
    return;

  const std::size_t n_frames = payload.size() / 64;
  std::int32_t x0 = rd_i32_be(&payload[4]);
  std::vector<std::int32_t> diffs;
  diffs.reserve(num_samples + 8);

  for (std::size_t f = 0; f < n_frames && diffs.size() < num_samples; ++f) {
    const std::uint8_t *fp = &payload[f * 64];
    const std::uint32_t ctrl = rd_u32_be(fp);

    for (int w = 1; w < 16 && diffs.size() < num_samples; ++w) {
      if (f == 0 && (w == 1 || w == 2))
        continue;
      const std::uint32_t ck = (ctrl >> ((15 - w) * 2)) & 0x3u;
      const std::uint32_t word = rd_u32_be(fp + w * 4);

      if (ck == 0u) {
        continue;
      } else if (ck == 1u) {
        diffs.push_back(
            static_cast<std::int32_t>(static_cast<std::int8_t>(word >> 24)));
        diffs.push_back(static_cast<std::int32_t>(
            static_cast<std::int8_t>((word >> 16) & 0xFFu)));
        diffs.push_back(static_cast<std::int32_t>(
            static_cast<std::int8_t>((word >> 8) & 0xFFu)));
        diffs.push_back(
            static_cast<std::int32_t>(static_cast<std::int8_t>(word & 0xFFu)));
      } else if (ck == 2u) {
        const std::uint32_t dnib = (word >> 30) & 0x3u;
        if (dnib == 1u) {
          diffs.push_back(sign_extend<30>(word));
        } else if (dnib == 2u) {
          diffs.push_back(sign_extend<15>(word >> 15));
          diffs.push_back(sign_extend<15>(word));
        } else if (dnib == 3u) {
          diffs.push_back(sign_extend<10>(word >> 20));
          diffs.push_back(sign_extend<10>(word >> 10));
          diffs.push_back(sign_extend<10>(word));
        }
      } else if (ck == 3u) {
        const std::uint32_t dnib = (word >> 30) & 0x3u;
        if (dnib == 0u) {
          for (int k = 4; k >= 0; --k)
            diffs.push_back(sign_extend<6>(word >> (k * 6)));
        } else if (dnib == 1u) {
          for (int k = 5; k >= 0; --k)
            diffs.push_back(sign_extend<5>(word >> (k * 5)));
        } else if (dnib == 2u) {
          for (int k = 6; k >= 0; --k)
            diffs.push_back(sign_extend<4>(word >> (k * 4)));
        }
      }
    }
  }

  if (diffs.empty())
    return;
  std::int32_t cur = x0;
  out.push_back(static_cast<double>(cur));
  for (std::size_t i = 1; i < num_samples && i < diffs.size(); ++i) {
    cur += diffs[i];
    out.push_back(static_cast<double>(cur));
  }
}

inline void decode_steim1_payload(std::span<const std::uint8_t> payload,
                                  std::size_t num_samples,
                                  std::vector<double> &out) {
  out.clear();
  out.reserve(num_samples);
  if (num_samples == 0 || payload.size() < 64)
    return;

  const std::size_t n_frames = payload.size() / 64;
  const std::int32_t x0 = rd_i32_be(&payload[4]);
  std::vector<std::int32_t> diffs;
  diffs.reserve(num_samples + 8);

  for (std::size_t f = 0; f < n_frames && diffs.size() < num_samples; ++f) {
    const std::uint8_t *fp = &payload[f * 64];
    const std::uint32_t ctrl = rd_u32_be(fp);

    for (int w = 1; w < 16 && diffs.size() < num_samples; ++w) {
      if (f == 0 && (w == 1 || w == 2))
        continue;
      const std::uint32_t ck = (ctrl >> ((15 - w) * 2)) & 0x3u;
      const std::uint8_t *wp = fp + w * 4;

      if (ck == 1u) {
        diffs.push_back(
            static_cast<std::int32_t>(static_cast<std::int8_t>(wp[0])));
        diffs.push_back(
            static_cast<std::int32_t>(static_cast<std::int8_t>(wp[1])));
        diffs.push_back(
            static_cast<std::int32_t>(static_cast<std::int8_t>(wp[2])));
        diffs.push_back(
            static_cast<std::int32_t>(static_cast<std::int8_t>(wp[3])));
      } else if (ck == 2u) {
        diffs.push_back(static_cast<std::int32_t>(rd_i16_be(wp)));
        diffs.push_back(static_cast<std::int32_t>(rd_i16_be(wp + 2)));
      } else if (ck == 3u) {
        diffs.push_back(rd_i32_be(wp));
      }
    }
  }

  if (diffs.empty())
    return;
  std::int32_t cur = x0;
  out.push_back(static_cast<double>(cur));
  for (std::size_t i = 1; i < num_samples && i < diffs.size(); ++i) {
    cur += diffs[i];
    out.push_back(static_cast<double>(cur));
  }
}

[[nodiscard]] inline bool decode_record(std::span<const std::uint8_t> rec,
                                        DecodedRecord &out) {
  if (!parse_record_header(rec, out.hdr))
    return false;
  if (out.hdr.data_offset < 48 || out.hdr.data_offset >= rec.size())
    return false;

  const auto payload = rec.subspan(out.hdr.data_offset);
  if (out.hdr.encoding == 10) {
    decode_steim1_payload(payload, out.hdr.num_samples, out.samples);
    return out.samples.size() == out.hdr.num_samples;
  }
  if (out.hdr.encoding == 11) {
    decode_steim2_payload(payload, out.hdr.num_samples, out.samples);
    return out.samples.size() == out.hdr.num_samples;
  }
  if (out.hdr.encoding == 1) {
    out.samples.clear();
    out.samples.reserve(out.hdr.num_samples);
    for (std::size_t i = 0;
         i < out.hdr.num_samples && (i + 1) * 2 <= payload.size(); ++i) {
      out.samples.push_back(static_cast<double>(rd_i16_be(&payload[i * 2])));
    }
    return out.samples.size() == out.hdr.num_samples;
  }
  if (out.hdr.encoding == 3) {
    out.samples.clear();
    out.samples.reserve(out.hdr.num_samples);
    for (std::size_t i = 0;
         i < out.hdr.num_samples && (i + 1) * 4 <= payload.size(); ++i) {
      out.samples.push_back(static_cast<double>(rd_i32_be(&payload[i * 4])));
    }
    return out.samples.size() == out.hdr.num_samples;
  }
  return false;
}

// Appends all 512-byte records from an in-memory buffer whose time span
// intersects [min_mjd, max_mjd] and whose channel matches `chan_filter`
// (or all channels if empty).
inline void decode_buffer_records(std::span<const std::uint8_t> buf,
                                  double min_mjd, double max_mjd,
                                  std::string_view chan_filter,
                                  std::vector<DecodedRecord> &records) {
  const std::size_t n_recs = buf.size() / kRecordBytes;
  for (std::size_t r = 0; r < n_recs; ++r) {
    const auto span = buf.subspan(r * kRecordBytes, kRecordBytes);
    RecordHeader hdr;
    if (!parse_record_header(span, hdr) || hdr.sample_rate_hz <= 0.0 ||
        hdr.num_samples == 0)
      continue;
    if (!chan_filter.empty() && hdr.channel != chan_filter)
      continue;
    const double dur_days =
        (static_cast<double>(hdr.num_samples) / hdr.sample_rate_hz) / 86400.0;
    const double end_mjd = hdr.start_mjd + dur_days;
    if (end_mjd < min_mjd || hdr.start_mjd > max_mjd)
      continue;

    DecodedRecord rec;
    if (decode_record(span, rec)) {
      records.push_back(std::move(rec));
    }
  }
}

// Sorts decoded records by start time and merges contiguous records into
// ContinuousTrace segments (splitting if a gap/overlap exceeds 1.5 samples).
[[nodiscard]] inline std::vector<ContinuousTrace>
assemble_traces(std::vector<DecodedRecord> &records) {
  std::vector<ContinuousTrace> traces;
  if (records.empty())
    return traces;

  std::ranges::sort(records,
                    [](const DecodedRecord &a, const DecodedRecord &b) {
                      if (a.hdr.channel != b.hdr.channel)
                        return a.hdr.channel < b.hdr.channel;
                      return a.hdr.start_epoch_s < b.hdr.start_epoch_s;
                    });

  for (const auto &rec : records) {
    const double dt = 1.0 / rec.hdr.sample_rate_hz;
    if (traces.empty() || traces.back().channel != rec.hdr.channel ||
        std::abs(traces.back().sample_rate_hz - rec.hdr.sample_rate_hz) >
            1e-3) {
      ContinuousTrace tr;
      tr.network = rec.hdr.network;
      tr.station = rec.hdr.station;
      tr.channel = rec.hdr.channel;
      tr.sample_rate_hz = rec.hdr.sample_rate_hz;
      tr.start_epoch_s = rec.hdr.start_epoch_s;
      tr.start_mjd = rec.hdr.start_mjd;
      tr.data = rec.samples;
      traces.push_back(std::move(tr));
      continue;
    }

    auto &cur = traces.back();
    const double expected_next_s =
        cur.start_epoch_s + static_cast<double>(cur.data.size()) * dt;
    const double gap_s = rec.hdr.start_epoch_s - expected_next_s;

    if (std::abs(gap_s) <= 1.5 * dt) {
      // Contiguous
      cur.data.insert(cur.data.end(), rec.samples.begin(), rec.samples.end());
    } else if (gap_s < -1.5 * dt) {
      // Overlapping duplicate record: append only non-overlapping tail if any
      const auto skip = static_cast<std::size_t>(std::llround(-gap_s / dt));
      if (skip < rec.samples.size()) {
        cur.data.insert(cur.data.end(),
                        rec.samples.begin() + static_cast<std::ptrdiff_t>(skip),
                        rec.samples.end());
      }
    } else {
      // Gap > 1.5 samples: start a new continuous segment
      ContinuousTrace tr;
      tr.network = rec.hdr.network;
      tr.station = rec.hdr.station;
      tr.channel = rec.hdr.channel;
      tr.sample_rate_hz = rec.hdr.sample_rate_hz;
      tr.start_epoch_s = rec.hdr.start_epoch_s;
      tr.start_mjd = rec.hdr.start_mjd;
      tr.data = rec.samples;
      traces.push_back(std::move(tr));
    }
  }
  return traces;
}

} // namespace chaosmap::io::mseed
