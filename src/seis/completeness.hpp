#pragma once

// Catalog magnitude of completeness.

#include <cmath>
#include <limits>
#include <map>
#include <span>

namespace chaosmap::seis {

// Maximum-curvature magnitude of completeness (Wiemer & Wyss 2000): the most
// populated `bin`-wide magnitude bin plus `correction` (0.2 compensates the
// method's known underestimate). NaN for an empty catalog.
[[nodiscard]] inline double max_curvature_mc(std::span<const double> mags,
                                             double bin = 0.1,
                                             double correction = 0.2) {
  std::map<long, std::size_t> counts;
  for (double m : mags) {
    if (std::isfinite(m))
      ++counts[std::lround(m / bin)];
  }
  if (counts.empty())
    return std::numeric_limits<double>::quiet_NaN();
  auto mode = counts.begin();
  for (auto it = counts.begin(); it != counts.end(); ++it) {
    if (it->second > mode->second)
      mode = it;
  }
  return static_cast<double>(mode->first) * bin + correction;
}

} // namespace chaosmap::seis
