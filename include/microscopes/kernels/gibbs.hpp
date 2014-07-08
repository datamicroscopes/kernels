#pragma once

#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/common/random_fwd.hpp>
#include <microscopes/mixture/model.hpp>

#include <vector>
#include <utility>

namespace microscopes {
namespace kernels {

struct gibbs {
  typedef std::vector<std::pair<const models::model *, float>> grid_t;

  static void
  hp(mixture::state &state,
     const std::vector<std::pair<size_t, grid_t>> &params,
     common::rng_t &rng);

  static void
  assign(mixture::state &state, common::recarray::dataview &view, common::rng_t &rng);

  static void
  assign_resample(mixture::state &state, common::recarray::dataview &view,
      size_t m, common::rng_t &rng);
};

} // namespace kernels
} // namespace microscopes
