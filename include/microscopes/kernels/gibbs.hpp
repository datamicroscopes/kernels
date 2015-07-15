#pragma once

#include <microscopes/common/random_fwd.hpp>
#include <microscopes/common/entity_state.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>

#include <vector>
#include <utility>

namespace microscopes {
namespace kernels {

struct gibbs {
    typedef std::vector<std::pair<const models::hypers *, float>> grid_t;

    static void
    assign(common::entity_based_state_object &state, common::rng_t &rng);

    static void
    assign_resample(common::entity_based_state_object &state, size_t m, common::rng_t &rng);

    static void
    hp(common::entity_based_state_object &state,
       const std::vector<std::pair<size_t, grid_t>> &params,
       common::rng_t &rng);

    static void
    perftest(common::entity_based_state_object &state,
             common::rng_t &rng);
};

} // namespace kernels
} // namespace microscopes
