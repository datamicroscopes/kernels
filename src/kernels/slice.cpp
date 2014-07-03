#include <microscopes/kernels/slice.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::mixture;
using namespace microscopes::kernels;
using namespace microscopes::models;

struct feature_scorefn {
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

struct cluster_scorefn {
  inline float
  operator()(float m)
  {
    *hp_ = m;
    return prior_scorefn_(m) + s_->score_assignment();
  }
  state *s_;
  scalar_1d_float_fn prior_scorefn_;
  float *hp_;
};

void
slice::hp(state &s,
          const vector<slice_hp_param_t> &cparams,
          const vector<slice_hp_t> &hparams,
          rng_t &rng)
{
  vector<size_t> indices;

  // slice on the feature HPs
  feature_scorefn feature_func;
  feature_func.s_ = &s;
  feature_func.rng_ = &rng;
  for (const auto &p : hparams) {
    feature_func.i_ = p.index_;
    util::permute(indices, p.params_.size(), rng);
    for (auto pi : indices) {
      const auto &p1 = p.params_[pi];
      // XXX: need some sort of runtime type checking here
      float *const px = reinterpret_cast<float *>(
          s.get_feature_hp_raw_ptr(p.index_, p1.key_)); // cringe
      MICROSCOPES_ASSERT(px);
      for (const auto &p2 : p1.components_) {
        // XXX: need some sort of bounds checking here
        float *const this_px = px + p2.index_;
        feature_func.prior_scorefn_ = p2.prior_;
        feature_func.hp_ = this_px;
        *this_px = sample(feature_func, *this_px, p2.w_, rng);
      }
    }
  }

  // slice on the cluster HPs
  cluster_scorefn cluster_func;
  cluster_func.s_ = &s;
  for (const auto &p : cparams) {
    float *const px = reinterpret_cast<float *>(
        s.get_cluster_hp_raw_ptr(p.key_));
    MICROSCOPES_ASSERT(px);
    for (const auto &p1 : p.components_) {
      float *const this_px = px + p1.index_;
      cluster_func.prior_scorefn_ = p1.prior_;
      cluster_func.hp_ = this_px;
      *this_px = sample(cluster_func, *this_px, p1.w_, rng);
    }
  }
}

struct theta_scorefn {
  inline float
  operator()(float m)
  {
    *hp_ = m;
    return s_->score_data({fi_}, {gi_}, *rng_);
  } state *s_;
  rng_t *rng_;
  size_t fi_;
  size_t gi_;
  float *hp_;
};

void
slice::theta(state &s,
             const vector<slice_theta_t> &tparams,
             rng_t &rng)
{
  vector<size_t> indices;
  const vector<size_t> groups = s.groups();
  theta_scorefn theta_func;
  theta_func.s_ = &s;
  theta_func.rng_ = &rng;
  for (const auto &p : tparams) {
    theta_func.fi_ = p.index_;
    for (const auto &p1 : p.params_) {
      util::permute(indices, groups.size(), rng);
      for (auto pi : indices) {
        float *const px = reinterpret_cast<float *>(
            s.get_suff_stats_raw_ptr(groups[pi], p.index_, p1.key_));
        MICROSCOPES_ASSERT(px);
        theta_func.hp_ = px;
        theta_func.gi_ = groups[pi];
        *px = sample(theta_func, *px, p1.w_, rng);
      }
    }
  }
}
