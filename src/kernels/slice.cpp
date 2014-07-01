#include <microscopes/kernels/slice.hpp>
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
    return prior_scorefn_(m) + s_->score_data({i_}, *rng_);
  }
  state *s_;
  rng_t *rng_;
  size_t i_;
  std::function<float(float)> prior_scorefn_;
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
      float *px = reinterpret_cast<float *>(s.get_feature_hp_raw_ptr(fid, kv.first)); // cringe
      assert(px);
      func.prior_scorefn_ = kv.second.first;
      func.hp_ = px;
      *px = sample(func, *px, kv.second.second, rng);
    }
  }
}
