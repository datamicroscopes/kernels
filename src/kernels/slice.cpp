#include <microscopes/kernels/slice.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::kernels;
using namespace microscopes::models;

struct feature_scorefn {
  inline float
  operator()(float m)
  {
    mut_->set<float>(m, pos_);
    return prior_scorefn_(m) + s_->score_likelihood(feature_, *rng_);
  }
  value_mutator *mut_;
  size_t pos_;
  fixed_entity_based_state_object *s_;
  rng_t *rng_;
  size_t feature_;
  scalar_1d_float_fn prior_scorefn_;
};

struct cluster_scorefn {
  inline float
  operator()(float m)
  {
    mut_->set<float>(m, pos_);
    return prior_scorefn_(m) + s_->score_assignment();
  }
  value_mutator *mut_;
  size_t pos_;
  fixed_entity_based_state_object *s_;
  scalar_1d_float_fn prior_scorefn_;
};

void
slice::hp(fixed_entity_based_state_object &s,
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
    feature_func.feature_ = p.index_;
    util::inplace_permute(indices, p.params_.size(), rng);
    for (auto pi : indices) {
      const auto &p1 = p.params_[pi];
      value_mutator mut = s.get_component_hp_mutator(p.index_, p1.key_);
      feature_func.mut_ = &mut;
      MICROSCOPES_DCHECK(
          mut.type().t() == TYPE_F32 ||
          mut.type().t() == TYPE_F64, "need floats");
      for (const auto &p2 : p1.components_) {
        MICROSCOPES_DCHECK(p2.index_ < mut.shape(), "index OOB");
        feature_func.pos_ = p2.index_;
        feature_func.prior_scorefn_ = p2.prior_;
        const float start = mut.accessor().get<float>(p2.index_);
        mut.set<float>(sample(feature_func, start, p2.w_, rng), p2.index_);
      }
    }
  }

  // slice on the cluster HPs
  cluster_scorefn cluster_func;
  cluster_func.s_ = &s;
  for (const auto &p : cparams) {
    value_mutator mut = s.get_cluster_hp_mutator(p.key_);
    cluster_func.mut_ = &mut;
    MICROSCOPES_DCHECK(
        mut.type().t() == TYPE_F32 ||
        mut.type().t() == TYPE_F64, "need floats");
    for (const auto &p1 : p.components_) {
      MICROSCOPES_DCHECK(p1.index_ < mut.shape(), "index OOB");
      cluster_func.pos_ = p1.index_;
      cluster_func.prior_scorefn_ = p1.prior_;
      const float start = mut.accessor().get<float>(p1.index_);
      mut.set<float>(sample(cluster_func, start, p1.w_, rng), p1.index_);
    }
  }
}

struct theta_scorefn {
  inline float
  operator()(float m)
  {
    mut_->set<float>(m, 0);
    return s_->score_likelihood(component_, id_, *rng_);
  }
  value_mutator *mut_;
  fixed_entity_based_state_object *s_;
  rng_t *rng_;
  size_t component_;
  ident_t id_;
};

void
slice::theta(fixed_entity_based_state_object &s,
             const vector<slice_theta_t> &tparams,
             rng_t &rng)
{
  vector<size_t> indices;
  theta_scorefn theta_func;
  theta_func.s_ = &s;
  theta_func.rng_ = &rng;
  for (const auto &p : tparams) {
    theta_func.component_ = p.index_;
    const auto idents = s.suffstats_identifiers(p.index_);
    for (const auto &p1 : p.params_) {
      util::inplace_permute(indices, idents.size(), rng);
      for (auto pi : indices) {
        value_mutator mut = s.get_suffstats_mutator(p.index_, idents[pi], p1.key_);
        theta_func.mut_ = &mut;
        MICROSCOPES_DCHECK(
            mut.type().t() == TYPE_F32 ||
            mut.type().t() == TYPE_F64, "need floats");
        MICROSCOPES_DCHECK(mut.shape() == 1, "assuming scalar parameter");
        theta_func.id_ = idents[pi];
        const float start = mut.accessor().get<float>(0);
        mut.set<float>(sample(theta_func, start, p1.w_, rng), 0);
      }
    }
  }
}
