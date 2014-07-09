#pragma once

#include <microscopes/common/random_fwd.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/common/entity_state.hpp>

#include <distributions/random.hpp>

#include <cmath>
#include <map>
#include <vector>
#include <utility>
#include <iostream>

namespace microscopes {
namespace kernels {

struct slice {

  template <typename T>
  static inline std::pair<float, float>
  interval(T fn, float x0, float y, float w, common::rng_t &rng, unsigned m)
  {
    const float U = distributions::sample_unif01(rng);
    float L = x0 - w*U;
    float R = L + w;
    const float V = distributions::sample_unif01(rng);
    const unsigned J = floor(m*V);
    const unsigned K = m-1-J;

    unsigned J0 = J;
    unsigned K0 = K;

    while (J0 > 0 && y < fn(L)) {
      L -= w;
      J0--;
    }

    while (K0 > 0 && y < fn(R)) {
      R += w;
      K0--;
    }

    if ((!J0 && J) || (!K0 && K)) {
      std::cout << "WARNING: slice::interval hit maximum # of expansions" << std::endl
                << "  Left expansions: " << J0 << ", Right expansions: " << K0 << std::endl
                << "  x0=" << x0 << ", y=" << y << ", w=" << w << std::endl
                << "  L=" << L << ", R=" << R << " fn(L)=" << fn(L) << ", fn(R)=" << fn(R) << std::endl;
    }

    return std::make_pair(L, R);
  }

  template <typename T>
  static inline float
  shrink(T fn, float x0, float y, float L, float R, common::rng_t &rng, unsigned ntries)
  {
    float x1 = 0.0;
    while (ntries) {
      const float U = distributions::sample_unif01(rng);
      x1 = L + U*(R-L);
      if (y < fn(x1))
        break;
      if (x1 < x0)
        L = x1;
      else
        R = x1;
      ntries--;
    }

    if (!ntries)
      std::cout << "WARNING: slice::shrink hit maximum # of iterations" << std::endl;

    //std::cout << "slice::shrink():" << std::endl
    //          << "  x0=" << x0 << ", y=" << y << ", L=" << L << ", R=" << R << std::endl
    //          << "  returning x1=" << x1 << ", fn(x1)=" << fn(x1)
    //          << std::endl;

    return x1;
  }

  template <typename T>
  static inline float
  sample(T scorefn, float x0, float w, common::rng_t &rng, unsigned m=10000, unsigned ntries=100)
  {
    const float y = logf(distributions::sample_unif01(rng)) + scorefn(x0);
    const auto p = interval(scorefn, x0, y, w, rng, m);
    return shrink(scorefn, x0, y, p.first, p.second, rng, ntries);
  }

  struct slice_hp_param_component_t {
    slice_hp_param_component_t() : index_(), prior_(), w_() {}
    slice_hp_param_component_t(
        size_t index,
        common::scalar_1d_float_fn prior,
        float w)
      : index_(index), prior_(prior), w_(w) {}

    size_t index_;
    common::scalar_1d_float_fn prior_;
    float w_;
  };

  struct slice_hp_param_t {
    slice_hp_param_t() : key_(), components_() {}
    slice_hp_param_t(
        const std::string &key,
        const std::vector<slice_hp_param_component_t> &components)
      : key_(key), components_(components) {}

    std::string key_;
    std::vector<slice_hp_param_component_t> components_;
  };

  struct slice_hp_t {
    slice_hp_t() : index_(), params_() {}
    slice_hp_t(
        size_t index,
        const std::vector<slice_hp_param_t> &params)
      : index_(index), params_(params) {}

    size_t index_;
    std::vector<slice_hp_param_t> params_;
  };

  static void
  hp(common::fixed_entity_based_state_object &state,
     const std::vector<slice_hp_param_t> &cparams,
     const std::vector<slice_hp_t> &hparams,
     common::rng_t &rng);

  struct slice_theta_param_t {
    slice_theta_param_t() : key_(), prior_(), w_() {}
    slice_theta_param_t(
        const std::string &key,
        float w)
      : key_(key), w_(w) {}

    std::string key_;
    common::scalar_1d_float_fn prior_;
    float w_;
  };

  struct slice_theta_t {
    slice_theta_t() : index_(), params_() {}
    slice_theta_t(
        size_t index,
        const std::vector<slice_theta_param_t> &params)
      : index_(index), params_(params) {}

    size_t index_;
    std::vector<slice_theta_param_t> params_;
  };

  static void
  theta(common::fixed_entity_based_state_object &state,
        const std::vector<slice_theta_t> &tparams,
        common::rng_t &rng);
};

} // namespace kernels
} // namespace microscopes
