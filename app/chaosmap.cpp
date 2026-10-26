#include "anomaly/array_anomaly.hpp"
#include "core/config.hpp"
#include "fusion/benchmark.hpp"
#include "fusion/fuse.hpp"
#include "geo/projection.hpp"
#include "gnss/fields.hpp"
#include "gnss/trajectory.hpp"
#include "io/catalog.hpp"
#include "io/tenv3.hpp"
#include "okada/inversion.hpp"
#include "out/export.hpp"
#include "seis/completeness.hpp"
#include "seis/planes.hpp"
#include "study/lseis_study.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Prints a fitted seismicity dip, which is NaN when the catalog depths cannot
// resolve it.
struct DipText {
  double deg;
};
std::ostream &operator<<(std::ostream &os, DipText d) {
  return std::isfinite(d.deg) ? (os << d.deg << " deg") : (os << "unresolved");
}

void print_usage() {
  std::cout << "Usage:\n"
            << "  chaosmap [all|gnss|invert|seismicity|lyapunov|fuse] --config "
               "<case.toml>\n"
            << "  chaosmap lseis-study --config <case.toml> [--surrogates N]\n"
            << "      [--pre-days D] [--post-days D] [--mask-mag M|mc] "
               "[--tag NAME]\n"
            << "      Surrogate-controlled test of the L-seis change across "
               "the mainshock.\n"
            << "      --mask-mag mc masks every catalog event at or above "
               "the pre-event\n"
            << "      magnitude of completeness; --tag suffixes the output "
               "file names.\n";
}

struct StudyOptions {
  chaosmap::study::LseisStudyParams params;
  std::string mask_mag; // number, "mc", or empty for the default
  std::string tag;
};

// `lseis-study`: does the post-mainshock change of the broadband Wolf
// exponent survive IAAFT surrogates and spectral covariates?
int run_lseis_study(const chaosmap::core::CaseConfig &cfg, StudyOptions opt) {
  using namespace chaosmap;
  auto &params = opt.params;
  // Every catalogued magnitude, so the mask can go below the pipeline's 2.5
  const auto events = io::read_catalog_csv(
      cfg.paths.catalog_csv, cfg.region.lat_min, cfg.region.lat_max,
      cfg.region.lon_min, cfg.region.lon_max, -10.0);

  // Magnitude of completeness before and after the mainshock over the span
  const auto origin_s = static_cast<double>(cfg.event.origin_epoch_s);
  std::vector<double> mags_pre, mags_post;
  for (const auto &e : events) {
    const double d = (static_cast<double>(e.epoch_s) - origin_s) / 86400.0;
    if (d >= -params.pre_days && d < 0.0)
      mags_pre.push_back(e.magnitude);
    else if (d >= 0.0 && d <= params.post_days)
      mags_post.push_back(e.magnitude);
  }
  const double mc_pre = seis::max_curvature_mc(mags_pre);
  const double mc_post = seis::max_curvature_mc(mags_post);
  if (opt.mask_mag == "mc") {
    if (!std::isfinite(mc_pre)) {
      std::cout << "[lseis-study] no pre-event catalog events for Mc\n";
      return EXIT_FAILURE;
    }
    params.mask_min_mag = mc_pre;
  } else if (!opt.mask_mag.empty()) {
    params.mask_min_mag = std::stod(opt.mask_mag);
  }

  const auto res = study::run_lseis_study(cfg, events, params);
  if (res.stations.empty()) {
    std::cout << "[lseis-study] no broadband stations with waveforms in "
              << cfg.paths.broadband_dir << "\n";
    return EXIT_FAILURE;
  }
  out::export_lseis_study(cfg, res, opt.tag);

  auto effect = [](const study::Effect &e) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << std::showpos << e.estimate
       << std::noshowpos << " [" << e.ci_lo << ", " << e.ci_hi << "]";
    return os.str();
  };
  std::size_t n_masked = 0;
  for (const auto &e : events)
    n_masked += (e.magnitude >= params.mask_min_mag) ? 1 : 0;
  std::cout << std::fixed << std::setprecision(1)
            << "[lseis-study] span origin -" << params.pre_days << " d / +"
            << params.post_days << " d; catalog Mc pre=" << mc_pre
            << ", post=" << mc_post << "; masking M>=" << params.mask_min_mag
            << " (" << n_masked << " events in the region catalog)\n";
  std::cout << "[lseis-study] " << res.windows.size() << " windows, "
            << params.n_surrogates
            << " IAAFT surrogates each; post - pre means with 95% day-block "
               "bootstrap CIs\n";
  for (const auto &s : res.stations) {
    std::cout << std::fixed << std::setprecision(2) << "  [" << s.station
              << " " << s.channel << "] dist=" << s.dist_km
              << " km, n_pre=" << s.n_pre << ", n_post=" << s.n_post
              << ", nonlinear windows pre/post=" << s.frac_nonlinear_pre
              << "/" << s.frac_nonlinear_post << ", R2(surr)=" << s.r2_surr
              << ", R2(cov)=" << s.r2_covariates << "\n"
              << "      d_wolf      " << effect(s.delta_wolf) << "\n"
              << "      d_surr_mean " << effect(s.delta_surr_mean)
              << "  (predicted by spectrum + amplitudes)\n"
              << "      d_excess    " << effect(s.delta_excess)
              << "  (wolf - surrogate mean)\n"
              << "      d_wolf_adj  " << effect(s.delta_wolf_adj)
              << "  (controlling for mean period, band RMS)\n";
  }
  std::cout << "  -> " << cfg.paths.out_dir << "\n";
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  std::string subcmd = "all";
  std::string config_path;
  StudyOptions study_opt;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return EXIT_SUCCESS;
    }
    if (arg == "--config" || arg == "-c") {
      if (i + 1 < argc) {
        config_path = argv[++i];
      }
    } else if (arg == "--surrogates" && i + 1 < argc) {
      study_opt.params.n_surrogates = std::atoi(argv[++i]);
    } else if (arg == "--pre-days" && i + 1 < argc) {
      study_opt.params.pre_days = std::atof(argv[++i]);
    } else if (arg == "--post-days" && i + 1 < argc) {
      study_opt.params.post_days = std::atof(argv[++i]);
    } else if (arg == "--mask-mag" && i + 1 < argc) {
      study_opt.mask_mag = argv[++i];
    } else if (arg == "--tag" && i + 1 < argc) {
      study_opt.tag = argv[++i];
    } else if (!arg.starts_with("--") && config_path.empty() &&
               arg.ends_with(".toml")) {
      config_path = std::string(arg);
    } else if (!arg.starts_with("--")) {
      subcmd = std::string(arg);
    }
  }

  if (config_path.empty()) {
    print_usage();
    return EXIT_FAILURE;
  }

  using namespace chaosmap;
  const auto cfg = core::CaseConfig::load(config_path);
  const auto grid = geo::Grid2D::make(cfg.region.lat_min, cfg.region.lat_max,
                                      cfg.region.lon_min, cfg.region.lon_max,
                                      cfg.region.grid_step_deg);

  std::cout << "==============================================================="
               "=====\n"
            << " CHAOSMAP: " << cfg.event.name << " (" << cfg.event.date_iso
            << ", MJD " << std::fixed << std::setprecision(1) << cfg.event.mjd
            << ")\n"
            << "==============================================================="
               "=====\n";

  if (subcmd == "lseis-study")
    return run_lseis_study(cfg, study_opt);

  // 1. GNSS Station Loading & Trajectory Decomposition
  const auto steps =
      io::read_steps_file(cfg.paths.gnss_dir / "meta" / "steps.txt");
  const auto raw_gnss =
      io::load_regional_gnss(cfg.paths.gnss_dir, cfg.event.lat, cfg.event.lon,
                             cfg.region.max_gnss_dist_km);

  std::vector<gnss::StationMotion> gnss_models;
  gnss_models.reserve(raw_gnss.size());
  std::size_t n_vel = 0, n_co = 0;
  for (const auto &s : raw_gnss) {
    const auto st_steps = steps.contains(s.station)
                              ? steps.at(s.station)
                              : std::vector<io::StepEntry>{};
    auto m = gnss::analyze_station(s, st_steps, cfg.event.mjd, cfg.event.lat,
                                   cfg.event.lon);
    if (m.has_velocity)
      ++n_vel;
    if (m.has_coseismic)
      ++n_co;
    gnss_models.push_back(std::move(m));
  }
  std::cout << "[1/6] GNSS Trajectory Decomposition: " << gnss_models.size()
            << " regional stations (" << n_vel << " interseismic velocity, "
            << n_co << " coseismic step)\n";
  if (subcmd == "gnss")
    return EXIT_SUCCESS;

  // 2. Okada 1985 Fault Plane & Distributed Slip Inversion
  const geo::LocalProjection proj(cfg.event.lat, cfg.event.lon);
  const auto inv = okada::invert_fault(
      gnss_models, proj, cfg.event.mw_ref, cfg.event.depth_km,
      cfg.event.strike_hint_deg, cfg.event.dip_hint_deg,
      cfg.event.rake_hint_deg, cfg.event.is_strike_slip);
  std::cout << std::setprecision(2) << "[2/6] Okada 1985 Slip Inversion:\n"
            << "      Fault Plane: strike=" << inv.plane.strike_deg
            << " deg, dip=" << inv.plane.dip_deg
            << " deg, rake=" << inv.mean_rake_deg << " deg\n"
            << "      Dimensions : L=" << inv.plane.length_km
            << " km, W=" << inv.plane.width_km
            << " km, center_depth=" << inv.plane.depth_km << " km\n"
            << "      How Much It Moved: peak_slip=" << inv.max_slip_m
            << " m, Mw=" << inv.mw_geodetic << " (ref " << cfg.event.mw_ref
            << "), VR=" << inv.variance_reduction_pct << "%\n";
  if (subcmd == "invert")
    return EXIT_SUCCESS;

  // 3. Pre-Event & Post-Event Seismicity 3D Plane Fitting
  const auto all_events = io::read_catalog_csv(
      cfg.paths.catalog_csv, cfg.region.lat_min, cfg.region.lat_max,
      cfg.region.lon_min, cfg.region.lon_max, 2.5);
  std::vector<io::CatalogEvent> pre_events, post_events;
  for (const auto &e : all_events) {
    if (e.epoch_s < cfg.event.origin_epoch_s &&
        e.epoch_s >= cfg.event.origin_epoch_s - 5LL * 365 * 86400) {
      pre_events.push_back(e);
    } else if (e.epoch_s >= cfg.event.origin_epoch_s &&
               e.epoch_s <= cfg.event.origin_epoch_s + 120LL * 86400) {
      post_events.push_back(e);
    }
  }

  const auto pre_seis = seis::analyze_seismicity(pre_events, grid, 3);
  const auto post_seis = seis::analyze_seismicity(post_events, grid, 3);
  std::cout << "[3/6] 3D Seismicity Planes:\n"
            << "      Before Event (5 yr): n=" << pre_events.size()
            << ", strike=" << pre_seis.dominant_plane.strike_deg
            << " deg, dip=" << DipText{pre_seis.dominant_plane.dip_deg} << "\n"
            << "      After Event (120 d): n=" << post_events.size()
            << ", strike=" << post_seis.dominant_plane.strike_deg
            << " deg, dip=" << DipText{post_seis.dominant_plane.dip_deg}
            << ", L=" << post_seis.dominant_plane.length_km << " km\n";
  if (subcmd == "seismicity")
    return EXIT_SUCCESS;

  // 4. Geodetic Strain Rate & Lagrangian FTLE (`L-geo`)
  const auto geod = gnss::compute_strain_and_ftle(gnss_models, grid);
  std::cout << "[4/6] Geodetic Strain & Lagrangian FTLE (L-geo):\n"
            << "      FTLE Principal Ridge strike=" << geod.ridge_strike_deg
            << " deg\n";

  // 5. Broadband Seismic Lyapunov (`L-seis`) & Benchmark
  std::vector<io::CatalogEvent> window_events = pre_events;
  window_events.insert(window_events.end(), post_events.begin(),
                       post_events.end());
  const auto seis_anom =
      anomaly::analyze_broadband_array(cfg, window_events, grid);
  if (seis_anom.has_seismic) {
    std::cout << "[5/6] Broadband Seismic Lyapunov (L-seis): "
              << seis_anom.stations.size()
              << " stations, array_pre_AUC=" << seis_anom.array_pre_auc
              << " (null p=" << seis_anom.array_null_p_value
              << "), ridge_strike=" << seis_anom.ridge_strike_deg << " deg\n";
    for (const auto &st : seis_anom.stations) {
      std::cout << "      [" << st.station << "] dist=" << st.dist_to_hypo_km
                << " km, wolf_pre=" << st.pre_wolf_median
                << ", wolf_post=" << st.post_wolf_median
                << ", d_wolf=" << st.delta_wolf_post_minus_pre
                << ", max_z=" << st.pre_horizon_max_z << "\n";
    }
  } else {
    std::cout << "[5/6] Broadband Seismic Lyapunov (L-seis): skipped (no "
                 "waveform archive configured)\n";
  }

  const auto bm =
      fusion::benchmark_lyapunov_candidates(geod, seis_anom, inv, grid);
  std::cout << "      Benchmark Mode: " << bm.best_candidate
            << " (w_L_geo=" << bm.l_geo.assigned_weight
            << ", w_L_seis=" << bm.l_seis.assigned_weight << ")\n";
  if (subcmd == "lyapunov")
    return EXIT_SUCCESS;

  // 6. Fuse Before / Moved / After / Change & Export
  const auto fused = fusion::fuse_fault_states(
      grid, gnss_models, geod, inv, pre_seis, post_seis, seis_anom, bm);
  out::export_case_results(cfg, grid, gnss_models, geod, inv, pre_seis,
                           post_seis, seis_anom, bm, fused);

  std::cout << "[6/6] Fusion & Export -> " << cfg.paths.out_dir << "\n"
            << "      Max Station Displacement: " << fused.max_disp_station
            << " (" << fused.max_station_horiz_m * 1000.0 << " mm horiz, "
            << fused.max_station_up_m * 1000.0 << " mm up)\n"
            << "      Peak Fault Slip         : " << fused.peak_slip_m
            << " m (top-quartile mean " << fused.top_quartile_slip_m << " m)\n"
            << "      Before->After Strike Δ  : " << fused.strike_change_deg
            << " deg, Centroid Shift: " << fused.centroid_shift_km << " km\n"
            << "==============================================================="
               "=====\n";

  return EXIT_SUCCESS;
}
