#include <microscopes/kernels/slice.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::mixture;
using namespace microscopes::kernels;
using namespace microscopes::models;

struct scorefn {
  inline float
  operator()(float m)
  {
    *hp_ = m;
    return prior_scorefn_(m) + s_->score_data({i_}, {}, *rng_);
  }
  state *s_;
  rng_t *rng_;
  size_t i_;
  scalar_1d_float_fn prior_scorefn_;
  float *hp_;
};

void
slice::hp(state &s, const vector<pair<size_t, slice_t>> &params, rng_t &rng)
{
  scorefn func; func.s_ = &s; func.rng_ = &rng;
  for (auto &p : params) {
    const size_t fid = p.first;
    func.i_ = fid;
    const slice_t &sl = p.second;
    // XXX: permute the order of keys
    for (auto &kv : sl) {
      // XXX: need some sort of runtime type checking here
      float *const px = reinterpret_cast<float *>(s.get_feature_hp_raw_ptr(fid, kv.first)); // cringe
      MICROSCOPES_ASSERT(px);
      for (auto &indiv : kv.second) {
        // XXX: need some sort of bounds checking here
        float *const this_px = px + indiv.index_;
        func.prior_scorefn_ = indiv.prior_;
        func.hp_ = this_px;
        *this_px = sample(func, *this_px, indiv.w_, rng);
      }
    }
  }
}
