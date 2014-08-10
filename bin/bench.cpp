
#include <microscopes/kernels/gibbs.hpp>
#include <microscopes/irm/model.hpp>
#include <microscopes/models/distributions.hpp>
#include <microscopes/common/relation/dataview.hpp>

#include <distributions/models/bb.hpp>

#include <memory>

using namespace std;
using namespace microscopes;
using namespace distributions;

// XXX: support mixturemodel

static common::hyperparam_bag_t
crp_hp_messsage(float alpha)
{
  io::CRP m;
  m.set_alpha(alpha);
  return common::util::protobuf_to_string(m);
}

static common::hyperparam_bag_t
bb_hp_messsage(float alpha, float beta)
{
  distributions::protobuf::BetaBernoulli::Shared m;
  m.set_alpha(alpha);
  m.set_beta(beta);
  return common::util::protobuf_to_string(m);
}

static pair<
  shared_ptr<irm::state<4>>,
  vector<shared_ptr<common::relation::dataview>>
>
make_irm(size_t groups,
         size_t entities_per_group,
         size_t relations,
         common::rng_t &r)
{
  const size_t n = groups * entities_per_group;

  irm::relation_definition reldef({0, 0},
      make_shared<models::distributions_model<BetaBernoulli>>());
  vector<irm::relation_definition> reldefs(relations, reldef);

  irm::model_definition defn({n}, reldefs);

  vector<common::hyperparam_bag_t> cluster_inits({crp_hp_messsage(1.)});
  vector<common::hyperparam_bag_t> relation_inits({bb_hp_messsage(1., 1.)});

  vector<size_t> assignment0;
  for (size_t i = 0; i < groups; i++)
    for (size_t j = 0; j < entities_per_group; j++)
      assignment0.push_back(i);

  // memory leaks
  bool * data = new bool[n * n];
  for (size_t i = 0; i < (n * n); i++)
    data[i] = bernoulli_distribution()(r);

  auto view = shared_ptr<common::relation::row_major_dense_dataview>(
      new common::relation::row_major_dense_dataview(
        reinterpret_cast<const uint8_t *>(data),
        nullptr,
        {n, n},
        common::runtime_type(TYPE_B)));

  auto latent = irm::state<4>::initialize(
      defn,
      cluster_inits,
      relation_inits,
      {assignment0},
      {view.get()},
      r);

  vector<shared_ptr<common::relation::dataview>> dataset({view});

  return make_pair(latent, dataset);
}

int
main(int argc, char **argv)
{
  common::rng_t r;
  auto p = make_irm(100, 100, 1, r);
  irm::model<4> m(p.first, 0, p.second);

  for (;;)
    kernels::gibbs::perftest(m, r);

  return 0;
}
