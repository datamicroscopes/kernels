#pragma once

#include <microscopes/common/dataview.hpp>
#include <microscopes/common/random_fwd.hpp>
#include <microscopes/mixture/model.hpp>

namespace microscopes {
namespace kernels {

struct gibbs {
  static void
  assign(mixture::state &state, common::dataview &view, common::rng_t &rng);
};

} // namespace kernels
} // namespace microscopes
