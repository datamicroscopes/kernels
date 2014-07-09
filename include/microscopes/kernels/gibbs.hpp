#pragma once

#include <microscopes/common/random_fwd.hpp>
#include <microscopes/common/entity_state.hpp>

#include <vector>
#include <utility>

namespace microscopes {
namespace kernels {

struct gibbs {
  typedef std::vector<std::pair<const models::model *, float>> grid_t;

  static void
  assign_fixed(common::fixed_entity_based_state_object &s, common::rng_t &rng);

  static void
  assign(common::entity_based_state_object &state, common::rng_t &rng);

  static void
  assign_resample(common::entity_based_state_object &state, size_t m, common::rng_t &rng);

  static void
  hp(common::fixed_entity_based_state_object &state,
     const std::vector<std::pair<size_t, grid_t>> &params,
     common::rng_t &rng);
};

} // namespace kernels
} // namespace microscopes
