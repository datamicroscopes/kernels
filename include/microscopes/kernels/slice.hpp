#pragma once

#include <microscopes/common/dataview.hpp>
#include <microscopes/common/random_fwd.hpp>
#include <microscopes/mixture/model.hpp>

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
    unsigned J = floor(m*V);
    unsigned K = m-1-J;

    while (J > 0 && y < fn(L)) {
      L -= w;
      J--;
    }

    while (K > 0 && y < fn(R)) {
      R += w;
      K--;
    }

    if (!J || !K)
      std::cout << "WARNING: slice::interval hit maximum # of expansions" << std::endl;

    //std::cout << "slice::interval():" << std::endl
    //          << "  x0=" << x0 << ", y=" << y << ", w=" << w << std::endl
    //          << "  L=" << L << ", R=" << R << " fn(L)=" << fn(L) << ", fn(R)=" << fn(R)
    //          << std::endl;

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
  sample(T scorefn, float x0, float w, common::rng_t &rng, unsigned m=1000, unsigned ntries=100)
  {
    const float y = logf(distributions::sample_unif01(rng)) + scorefn(x0);
    const auto p = interval(scorefn, x0, y, w, rng, m);
    return shrink(scorefn, x0, y, p.first, p.second, rng, ntries);
  }

  typedef std::map<
      std::string,
      std::pair<std::function<float(float)>, float>
    > slice_t;

  static void
  hp(mixture::state &state,
     const std::vector<std::pair<size_t, slice_t>> &params,
     common::rng_t &rng);
};

} // namespace kernels
} // namespace microscopes
