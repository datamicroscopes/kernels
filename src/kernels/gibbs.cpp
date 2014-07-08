#include <microscopes/kernels/gibbs.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::common::recarray;
using namespace microscopes::mixture;
using namespace microscopes::kernels;
using namespace microscopes::models;

void
gibbs::hp(state &s, const vector<pair<size_t, grid_t>> &params, rng_t &rng)
{
  vector<float> scores;
  for (auto &p : params) {
    const size_t fid = p.first;
    const vector<size_t> features({fid});
    const grid_t &g = p.second;
    scores.reserve(g.size());
    scores.clear();
    for (auto &g0 : g) {
      s.set_feature_hp(fid, *g0.first);
      scores.push_back(g0.second + s.score_data(features, {}, rng));
    }
    const auto choice = util::sample_discrete_log(scores, rng);
    s.set_feature_hp(fid, *g[choice].first);
  }
}

void
gibbs::assign(state &s, dataview &view, rng_t &rng)
{
  // ensure 1 empty group
  // note: this is more efficient than s.ensure_k_empty_groups(1, rng)
  size_t egid = 0;
  const size_t egsizeinit = s.empty_groups().size();
  if (!egsizeinit)
    egid = s.create_group(rng);
  else {
    auto it = s.empty_groups().begin();
    egid = *it++;
    if (egsizeinit > 1) {
      vector<size_t> egremove(it, s.empty_groups().end());
      for (auto g : egremove)
        s.delete_group(g);
    }
  }

#ifdef DEBUG_MODE
  s.dcheck_consistency();
#endif
  for (view.reset(); !view.end(); view.next()) {
    const size_t gid = s.remove_value(view, rng);
    if (!s.groupsize(gid))
      s.delete_group(gid);
    MICROSCOPES_ASSERT(s.empty_groups().size() == 1);
    row_accessor acc = view.get();
    auto scores = s.score_value(acc, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    s.add_value(choice, view, rng);
    if (choice == egid) {
      egid = s.create_group(rng);
    }
#ifdef DEBUG_MODE
    s.dcheck_consistency();
#endif
  }
}

void
gibbs::assign_resample(state &s, dataview &view, size_t m, rng_t &rng)
{
#ifdef DEBUG_MODE
  s.dcheck_consistency();
#endif
  for (view.reset(); !view.end(); view.next()) {
    s.remove_value(view, rng);
    s.ensure_k_empty_groups(m, true, rng);
    row_accessor acc = view.get();
    auto scores = s.score_value(acc, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    s.add_value(choice, view, rng);
#ifdef DEBUG_MODE
    s.dcheck_consistency();
#endif
  }
}
