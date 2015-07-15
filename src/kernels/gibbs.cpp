#include <microscopes/kernels/gibbs.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::kernels;
using namespace microscopes::models;

static inline ALWAYS_INLINE void
AssertAllAssigned(const entity_based_state_object &state)
{
#ifdef DEBUG_MODE
  for (auto gid : state.assignments())
    MICROSCOPES_DCHECK(gid != -1, "unassigned entity found");
#endif
}

void
gibbs::assign(entity_based_state_object &state, rng_t &rng)
{
  AssertAllAssigned(state);
  pair<vector<size_t>, vector<float>> scores;
  // ensure 1 empty group
  const auto empty_groups = state.empty_groups();
  size_t egid = 0;
  if (empty_groups.empty()) {
    egid = state.create_group(rng);
  } else {
    auto it = empty_groups.begin();
    egid = *it++;
    for (; it != empty_groups.end(); ++it)
      state.delete_group(*it);
  }
  for (auto i : util::permute(state.nentities(), rng)) {
    const size_t gid = state.remove_value(i, rng);
    if (!state.groupsize(gid))
      state.delete_group(gid);
    MICROSCOPES_ASSERT(state.empty_groups().size() == 1);
    state.inplace_score_value(scores, i, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    state.add_value(choice, i, rng);
    if (choice == egid)
      egid = state.create_group(rng);
  }
}

void
gibbs::assign_resample(entity_based_state_object &state, size_t m, rng_t &rng)
{
  // Implements Algorithm 8 from:
  //   Markov Chain Sampling Methods for Dirichlet Process Mixture Models
  //   Radford Neal
  //   Journal of Computational and Graphical Statistics, 2000
  //   http://www.cs.toronto.edu/~radford/mixmc.abstract.html
  AssertAllAssigned(state);
  pair<vector<size_t>, vector<float>> scores;
  MICROSCOPES_DCHECK(m > 0, "need >=1 # of ephmeral groups");
  for (auto i : util::permute(state.nentities(), rng)) {
    const size_t gid = state.remove_value(i, rng);

    // delete all empty groups
    // [except if we created an empty group by calling remove_value()]
    bool match = false;
    for (auto g : state.empty_groups()) {
      if (g == gid) {
        // Alg 8 allows us to keep the value drawn from G_0 in this case:
        // "If c_i \neq c_j for all j \neq i, let c_i have the label k^- + 1,
        // and draw values independently from G_0 for those \phi_c for which
        // k^- + 1 < c \leq h."
        match = true;
        continue;
      }
      state.delete_group(g);
    }
    MICROSCOPES_ASSERT(state.empty_groups().size() == 0 ||
                       state.empty_groups().size() == 1);

    // create m new groups
    for (size_t g = (match ? 1 : 0); g < m; g++)
      state.create_group(rng);

    MICROSCOPES_ASSERT(state.empty_groups().size() == m);

    state.inplace_score_value(scores, i, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    state.add_value(choice, i, rng);
  }
}

void
gibbs::hp(entity_based_state_object &state,
          const vector<pair<size_t, grid_t>> &params,
          rng_t &rng)
{
  vector<float> scores;
  for (const auto &p : params) {
    const size_t fid = p.first;
    const grid_t &g = p.second;
    scores.reserve(g.size());
    scores.clear();
    for (const auto &g0 : g) {
      state.set_component_hp(fid, *g0.first);
      scores.push_back(g0.second + state.score_likelihood(fid, rng));
    }
    const auto choice = util::sample_discrete_log(scores, rng);
    state.set_component_hp(fid, *g[choice].first);
  }
}

// for performance debugging purposes
// doesn't change the group assignments
void
gibbs::perftest(entity_based_state_object &state, rng_t &rng)
{
  AssertAllAssigned(state);
  pair<vector<size_t>, vector<float>> scores;
  for (auto i : util::permute(state.nentities(), rng)) {
    const size_t gid = state.remove_value(i, rng);
    state.inplace_score_value(scores, i, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    (void)choice; // XXX: make sure compiler does not optimize this out
    state.add_value(gid, i, rng);
  }
}
