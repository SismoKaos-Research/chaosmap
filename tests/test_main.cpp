#include "chaos/lyapunov.hpp"
#include "core/config.hpp"
#include "core/linalg.hpp"
#include "dsp/filter.hpp"
#include "geo/projection.hpp"
#include "gnss/fields.hpp"
#include "gnss/trajectory.hpp"
#include "io/catalog.hpp"
#include "io/tenv3.hpp"
#include "io/zip_mseed.hpp"
#include "okada/inversion.hpp"
#include "okada/okada85.hpp"
#include "seis/planes.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

int g_failures = 0;

void check(bool cond, const char *expr, const char *file, int line) {
  if (!cond) {
    std::cerr << "[FAIL] " << file << ":" << line << " (" << expr << ")\n";
    ++g_failures;
  }
}

void check_near(double a, double b, double tol, const char *expr,
                const char *file, int line) {
  if (!(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol)) {
    std::cerr << "[FAIL] " << file << ":" << line << " (" << expr << ") got "
              << a << " vs " << b << " (tol " << tol << ")\n";
    ++g_failures;
  }
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol)                                                  \
  check_near((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)

void test_linalg_and_geo() {
  using namespace chaosmap::linalg;
  using namespace chaosmap::geo;

  // 1. WLS on y = 2.0 + 3.5 * x
  Mat A(5, 2);
  std::vector<double> b(5), sig(5, 0.1);
  for (std::size_t i = 0; i < 5; ++i) {
    const double x = static_cast<double>(i);
    A(i, 0) = 1.0;
    A(i, 1) = x;
    b[i] = 2.0 + 3.5 * x;
  }
  auto res = solve_wls(A, b, sig);
  CHECK_NEAR(res.x[0], 2.0, 1e-10);
  CHECK_NEAR(res.x[1], 3.5, 1e-10);
  CHECK_NEAR(res.rss, 0.0, 1e-12);

  // 2. 2x2 and 3x3 symmetric eigensolvers
  auto e2 = sym_eig2(3.0, 1.0, 3.0);
  CHECK_NEAR(e2.values[0], 2.0, 1e-12);
  CHECK_NEAR(e2.values[1], 4.0, 1e-12);

  auto e3 = sym_eig3({2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0});
  CHECK_NEAR(e3.values[0], 1.0, 1e-12);
  CHECK_NEAR(e3.values[1], 2.0, 1e-12);
  CHECK_NEAR(e3.values[2], 5.0, 1e-12);

  // 3. LocalProjection round-trip
  LocalProjection proj(38.721, 43.508);
  auto enu = proj.to_enu(39.0, 44.0);
  auto ll = proj.to_latlon(enu.x, enu.y);
  CHECK_NEAR(ll.x, 39.0, 1e-6);
  CHECK_NEAR(ll.y, 44.0, 1e-6);
  const double d_hav = haversine_km(38.721, 43.508, 39.0, 44.0);
  CHECK_NEAR(enu.norm(), d_hav, 0.05);
}

void test_gnss_van_and_kahramanmaras() {
  using namespace chaosmap;
  const auto van_cfg = core::CaseConfig::load("configs/van2011.toml");
  CHECK(van_cfg.event.id == "van2011");
  CHECK_NEAR(van_cfg.event.mjd, 55857.0, 1e-6);

  const auto steps =
      io::read_steps_file(van_cfg.paths.gnss_dir / "meta" / "steps.txt");
  CHECK(!steps.empty());

  // Check MRDY on Van 2011 (37 km from epicenter, ~57 mm southward motion)
  const auto mrdy =
      io::read_tenv3(van_cfg.paths.gnss_dir / "IGS20" / "MRDY.tenv3");
  CHECK(!mrdy.samples.empty());
  const auto mrdy_steps =
      steps.contains("MRDY") ? steps.at("MRDY") : std::vector<io::StepEntry>{};
  const auto mrdy_mot =
      gnss::analyze_station(mrdy, mrdy_steps, van_cfg.event.mjd,
                            van_cfg.event.lat, van_cfg.event.lon);
  CHECK(mrdy_mot.has_coseismic);
  CHECK(mrdy_mot.has_velocity);
  // MRDY moved ~ -16 mm East, ~ -57 mm North during Van 2011
  CHECK_NEAR(mrdy_mot.coseismic_enu_m.x, -0.016, 0.006);
  CHECK_NEAR(mrdy_mot.coseismic_enu_m.y, -0.057, 0.008);

  // Check EKZ1 and KAHR on Kahramanmaras 2023
  const auto km_cfg = core::CaseConfig::load("configs/kahramanmaras2023.toml");
  const auto ekz1 =
      io::read_tenv3(km_cfg.paths.gnss_dir / "IGS20" / "EKZ1.tenv3");
  const auto ekz1_steps =
      steps.contains("EKZ1") ? steps.at("EKZ1") : std::vector<io::StepEntry>{};
  const auto ekz1_mot = gnss::analyze_station(
      ekz1, ekz1_steps, km_cfg.event.mjd, km_cfg.event.lat, km_cfg.event.lon);
  CHECK(ekz1_mot.has_coseismic);
  // EKZ1 moved ~ -4.65 m West, +0.49 m North during 2023-02-06 sequence
  CHECK_NEAR(ekz1_mot.coseismic_enu_m.x, -4.65, 0.05);
  CHECK_NEAR(ekz1_mot.coseismic_enu_m.y, 0.49, 0.05);
}

void test_okada85_table2() {
  using namespace chaosmap::okada;
  // Okada (1985) Table 2 (Case 2: x=2, y=3, d=4, L=3, W=2, lambda=mu =>
  // nu=0.25) 1) delta = 70 deg, Strike-slip (U1=1)
  auto s70 = okada85_local(2.0, 3.0, 4.0, 70.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.25);
  CHECK_NEAR(s70.x, -8.689e-3, 1e-5);
  CHECK_NEAR(s70.y, -4.298e-3, 1e-5);
  CHECK_NEAR(s70.z, -2.747e-3, 1e-5);

  // 2) delta = 70 deg, Dip-slip (U2=1)
  auto d70 = okada85_local(2.0, 3.0, 4.0, 70.0, 3.0, 2.0, 0.0, 1.0, 0.0, 0.25);
  CHECK_NEAR(d70.x, -4.682e-3, 1e-5);
  CHECK_NEAR(d70.y, -3.527e-2, 2e-5);
  CHECK_NEAR(d70.z, -3.564e-2, 2e-5);

  // 3) delta = 70 deg, Tensile (U3=1)
  auto t70 = okada85_local(2.0, 3.0, 4.0, 70.0, 3.0, 2.0, 0.0, 0.0, 1.0, 0.25);
  CHECK_NEAR(t70.x, -2.660e-4, 1e-6);
  CHECK_NEAR(t70.y, +1.056e-2, 1e-5);
  CHECK_NEAR(t70.z, +3.214e-3, 1e-5);

  // 4) delta = 90 deg (vertical fault singularity branch) continuity check
  // vs 89.999 deg
  auto s89 =
      okada85_local(2.0, 3.0, 4.0, 89.999, 3.0, 2.0, 1.0, 0.0, 0.0, 0.25);
  auto s90 = okada85_local(2.0, 3.0, 4.0, 90.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.25);
  CHECK_NEAR(s90.x, s89.x, 1e-6);
  CHECK_NEAR(s90.y, s89.y, 1e-6);
  CHECK_NEAR(s90.z, s89.z, 1e-6);

  auto d89 =
      okada85_local(2.0, 3.0, 4.0, 89.999, 3.0, 2.0, 0.0, 1.0, 0.0, 0.25);
  auto d90 = okada85_local(2.0, 3.0, 4.0, 90.0, 3.0, 2.0, 0.0, 1.0, 0.0, 0.25);
  CHECK_NEAR(d90.x, d89.x, 1e-6);
  CHECK_NEAR(d90.y, d89.y, 1e-6);
  CHECK_NEAR(d90.z, d89.z, 1e-6);
}

void test_fault_inversion() {
  using namespace chaosmap;
  for (const char *cfg_file :
       {"configs/van2011.toml", "configs/kahramanmaras2023.toml"}) {
    const auto cfg = core::CaseConfig::load(cfg_file);
    const auto steps =
        io::read_steps_file(cfg.paths.gnss_dir / "meta" / "steps.txt");
    const auto raw_sta =
        io::load_regional_gnss(cfg.paths.gnss_dir, cfg.event.lat, cfg.event.lon,
                               cfg.region.max_gnss_dist_km);
    std::vector<gnss::StationMotion> motions;
    for (const auto &s : raw_sta) {
      const auto st_steps = steps.contains(s.station)
                                ? steps.at(s.station)
                                : std::vector<io::StepEntry>{};
      motions.push_back(gnss::analyze_station(s, st_steps, cfg.event.mjd,
                                              cfg.event.lat, cfg.event.lon));
    }
    const geo::LocalProjection proj(cfg.event.lat, cfg.event.lon);
    const auto inv =
        okada::invert_fault(motions, proj, cfg.event.mw_ref, cfg.event.depth_km,
                            cfg.event.strike_hint_deg, cfg.event.dip_hint_deg,
                            cfg.event.rake_hint_deg, cfg.event.is_strike_slip);
    CHECK(!inv.patches.empty());
    CHECK(inv.max_slip_m > 0.3);
    CHECK(inv.mw_geodetic >= cfg.event.mw_ref - 0.6 &&
          inv.mw_geodetic <= cfg.event.mw_ref + 0.6);
    CHECK(inv.variance_reduction_pct > 50.0);
    std::cout << "  [" << cfg.event.id << "] strike=" << inv.plane.strike_deg
              << " dip=" << inv.plane.dip_deg << " max_slip=" << inv.max_slip_m
              << " m"
              << " Mw=" << inv.mw_geodetic
              << " VR=" << inv.variance_reduction_pct << "%\n";

    // Also test Phase 4: strain & FTLE on the regional grid
    const auto grid = geo::Grid2D::make(cfg.region.lat_min, cfg.region.lat_max,
                                        cfg.region.lon_min, cfg.region.lon_max,
                                        cfg.region.grid_step_deg);
    const auto fields = gnss::compute_strain_and_ftle(motions, grid);
    CHECK(fields.ftle_interseismic.size() == grid.size());
    CHECK(fields.ridge_strike_deg > 0.0);
  }
}

void test_catalog_and_seismicity_planes() {
  using namespace chaosmap;
  const auto cfg = core::CaseConfig::load("configs/van2011.toml");
  const auto events = io::read_catalog_csv(
      cfg.paths.catalog_csv, cfg.region.lat_min, cfg.region.lat_max,
      cfg.region.lon_min, cfg.region.lon_max, 3.0);
  CHECK(!events.empty());

  std::vector<io::CatalogEvent> pre_ev, post_ev;
  for (const auto &e : events) {
    if (e.epoch_s < cfg.event.origin_epoch_s &&
        e.epoch_s >= cfg.event.origin_epoch_s - 4LL * 365 * 86400) {
      pre_ev.push_back(e);
    } else if (e.epoch_s >= cfg.event.origin_epoch_s &&
               e.epoch_s <= cfg.event.origin_epoch_s + 45LL * 86400) {
      post_ev.push_back(e);
    }
  }
  CHECK(post_ev.size() > 50);
  const auto grid = geo::Grid2D::make(cfg.region.lat_min, cfg.region.lat_max,
                                      cfg.region.lon_min, cfg.region.lon_max,
                                      cfg.region.grid_step_deg);
  const auto post_seis = seis::analyze_seismicity(post_ev, grid);
  CHECK(post_seis.dominant_plane.length_km > 20.0);
  CHECK(post_seis.dominant_plane.planarity > 0.2);
  CHECK(post_seis.dominant_plane.dip_deg > 20.0);
  std::cout << "  [van2011 aftershocks] n=" << post_ev.size()
            << " strike=" << post_seis.dominant_plane.strike_deg
            << " dip=" << post_seis.dominant_plane.dip_deg
            << " len=" << post_seis.dominant_plane.length_km << " km\n";
}

void test_ftle_and_waveforms_lyapunov() {
  using namespace chaosmap;
  // Synthetic shear zone across y = 0 for FTLE ridge test
  const auto grid = geo::Grid2D::make(38.0, 39.0, 42.5, 44.0, 0.05);
  std::vector<gnss::StationMotion> st;
  for (int i = -2; i <= 2; ++i) {
    for (int j = -2; j <= 2; ++j) {
      gnss::StationMotion m;
      const auto ll = grid.proj.to_latlon(i * 25.0, j * 20.0);
      m.lat_deg = ll.x;
      m.lon_deg = ll.y;
      m.vel_enu_m_yr = {(j >= 0) ? 0.015 : -0.015, 0.0, 0.0};
      m.vel_sigma_m_yr = {0.001, 0.001, 0.002};
      m.has_velocity = true;
      st.push_back(m);
    }
  }
  const auto fields = gnss::compute_strain_and_ftle(st, grid);
  CHECK(fields.ftle_interseismic.size() == grid.size());
  CHECK(geo::haversine_km(fields.ridge_start_lat, fields.ridge_start_lon,
                          fields.ridge_end_lat, fields.ridge_end_lon) > 15.0);

  // Real Van 2011 broadband waveform ingest + Lyapunov test (station TVAN)
  const double mjd0 = 55849.0; // 2011-10-15 (start of TVAN_2011-10-15.zip)
  const auto traces = io::load_station_waveforms(
      "/home/oguzb/data/sismokaos/tdvms/van_raw/TVAN", mjd0, mjd0 + 0.02, "BH");
  CHECK(!traces.empty());
  if (!traces.empty()) {
    const auto &tr = traces.front();
    CHECK(tr.sample_rate_hz >= 20.0);
    const std::size_t win_samples =
        static_cast<std::size_t>(600.0 * tr.sample_rate_hz);
    CHECK(tr.data.size() >= win_samples);
    const auto filt = dsp::preprocess_window(
        std::span<const double>(tr.data.data(), win_samples), tr.sample_rate_hz,
        5.0, 0.05, 0.5);
    CHECK(filt.size() == 3000);
    const auto wf =
        chaos::extract_window_features(filt, 5.0, tr.start_mjd, 5, 5, 5);
    CHECK(std::isfinite(wf.wolf_lye));
    CHECK(std::isfinite(wf.ros_short));
    CHECK(std::isfinite(wf.samp_ent));
    CHECK(wf.corr_dim > 0.0);
    std::cout << "  [TVAN " << tr.channel << "] mp=" << wf.mean_period_s
              << "s wolf=" << wf.wolf_lye << " ros_s=" << wf.ros_short
              << " samp_ent=" << wf.samp_ent << " corr_dim=" << wf.corr_dim
              << "\n";
  }
}

} // namespace

int main() {
  test_linalg_and_geo();
  test_gnss_van_and_kahramanmaras();
  test_okada85_table2();
  test_fault_inversion();
  test_catalog_and_seismicity_planes();
  test_ftle_and_waveforms_lyapunov();
  if (g_failures == 0) {
    std::cout << "[PASS] All chaosmap tests passed.\n";
    return EXIT_SUCCESS;
  }
  std::cerr << "[FAIL] " << g_failures << " test assertion(s) failed.\n";
  return EXIT_FAILURE;
}
