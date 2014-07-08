#pragma once

#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/common/random_fwd.hpp>
#include <microscopes/mixture/model.hpp>

namespace microscopes {
namespace kernels {

struct bootstrap {
  static void
  likelihood(mixture::state &state, common::recarray::dataview &view, common::rng_t &rng);
};

} // namespace kernels
} // namespace microscopes

