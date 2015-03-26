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
    args_[argpos_] = m;
    return prior_scorefn_(args_) + s_->score_likelihood(feature_, *rng_);
  }
  value_mutator *mut_;
  size_t pos_;
  size_t argpos_;
  entity_based_state_object *s_;
  rng_t *rng_;
  size_t feature_;
  scalar_fn prior_scorefn_;
  vector<float> args_;
};

struct cluster_scorefn {
  inline float
  operator()(float m)
  {
    mut_->set<float>(m, pos_);
    args_[argpos_] = m;
    return prior_scorefn_(args_) + s_->score_assignment();
  }
  value_mutator *mut_;
  size_t pos_;
  size_t argpos_;
  entity_based_state_object *s_;
  scalar_fn prior_scorefn_;
  vector<float> args_;
};

void
slice::hp(entity_based_state_object &s,
          const vector<slice_hp_param_t> &cparams,
          const vector<slice_hp_t> &hparams,
          rng_t &rng)
{
  vector<size_t> indices;
  vector<value_mutator> mutators;

  // slice on the feature HPs
  feature_scorefn feature_func;
  feature_func.s_ = &s;
  feature_func.rng_ = &rng;
  for (const auto &p : hparams) { // XXX: permute the hparams?
    feature_func.feature_ = p.index_;
    util::inplace_permute(indices, p.params_.size(), rng);
    for (auto pi : indices) {
      const auto &p1 = p.params_[pi];
      feature_func.prior_scorefn_ = p1.prior_;
      feature_func.args_.clear();

      // bootstrap the args (and save the mutators)
      mutators.clear();
      for (const auto &update : p1.updates_) {
        mutators.emplace_back(s.get_component_hp_mutator(p.index_, update.key_));
        MICROSCOPES_DCHECK(update.index_ < mutators.back().shape(), "update index OOB");
        feature_func.args_.push_back(mutators.back().accessor().get<float>(update.index_));
      }

      // XXX: permute this order?
      for (size_t i = 0; i < p1.updates_.size(); i++) {
        value_mutator &mut = mutators[i];
        const size_t index = p1.updates_[i].index_;
        MICROSCOPES_DCHECK(
            mut.type().t() == TYPE_F32 || mut.type().t() == TYPE_F64,
            "need floats");
        feature_func.mut_ = &mut;
        feature_func.pos_ = index;
        feature_func.argpos_ = i;
        const float start = feature_func.args_[i];
        const float samp = sample(feature_func, start, p1.w_, rng);
        mut.set<float>(samp, index);
        feature_func.args_[i] = samp;
      }
    }
  }

  // XXX: fix the code duplication

  // slice on the cluster HPs
  cluster_scorefn cluster_func;
  cluster_func.s_ = &s;
  // XXX: permute the cparams?
  for (const auto &p : cparams) {
    cluster_func.prior_scorefn_ = p.prior_;
    cluster_func.args_.clear();

    mutators.clear();
    for (const auto &update : p.updates_) {
      mutators.emplace_back(s.get_cluster_hp_mutator(update.key_));
      MICROSCOPES_DCHECK(update.index_ < mutators.back().shape(), "update index OOB");
      cluster_func.args_.push_back(mutators.back().accessor().get<float>(update.index_));
    }

    // XXX: permute this order?
    for (size_t i = 0; i < p.updates_.size(); i++) {
      value_mutator &mut = mutators[i];
      const size_t index = p.updates_[i].index_;
      MICROSCOPES_DCHECK(
          mut.type().t() == TYPE_F32 || mut.type().t() == TYPE_F64,
          "need floats");
      cluster_func.mut_ = &mut;
      cluster_func.pos_ = index;
      cluster_func.argpos_ = i;
      const float start = cluster_func.args_[i];
      const float samp = sample(cluster_func, start, p.w_, rng);
      mut.set<float>(samp, index);
      cluster_func.args_[i] = samp;
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
  entity_based_state_object *s_;
  rng_t *rng_;
  size_t component_;
  ident_t id_;
};

void
slice::theta(entity_based_state_object &s,
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
