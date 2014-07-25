#include <microscopes/kernels/gibbs.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::kernels;
using namespace microscopes::models;

static inline ALWAYS_INLINE void
AssertAllAssigned(const fixed_entity_based_state_object &s)
{
#ifdef DEBUG_MODE
  for (auto gid : s.assignments())
    MICROSCOPES_DCHECK(gid != -1, "unassigned entity found");
#endif
}

void
gibbs::assign_fixed(fixed_entity_based_state_object &s, rng_t &rng)
{
  AssertAllAssigned(s);
  for (auto i : util::permute(s.nentities(), rng)) {
    s.remove_value(i, rng);
    auto scores = s.score_value(i, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    s.add_value(choice, i, rng);
  }
}

void
gibbs::assign(entity_based_state_object &s, rng_t &rng)
{
  AssertAllAssigned(s);
  // ensure 1 empty group
  const auto empty_groups = s.empty_groups();
  size_t egid = 0;
  if (empty_groups.empty()) {
    egid = s.create_group(rng);
  } else {
    auto it = empty_groups.begin();
    egid = *it++;
    for (; it != empty_groups.end(); ++it)
      s.delete_group(*it);
  }
  for (auto i : util::permute(s.nentities(), rng)) {
    const size_t gid = s.remove_value(i, rng);
    if (!s.groupsize(gid))
      s.delete_group(gid);
    MICROSCOPES_ASSERT(s.empty_groups().size() == 1);
    auto scores = s.score_value(i, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    s.add_value(choice, i, rng);
    if (choice == egid)
      egid = s.create_group(rng);
  }
}

void
gibbs::assign_resample(entity_based_state_object &s, size_t m, rng_t &rng)
{
  // Implements Algorithm 8 from:
  //   Markov Chain Sampling Methods for Dirichlet Process Mixture Models
  //   Radford Neal
  //   Journal of Computational and Graphical Statistics, 2000
  //   http://www.cs.toronto.edu/~radford/mixmc.abstract.html
  AssertAllAssigned(s);
  MICROSCOPES_DCHECK(m > 0, "need >=1 # of ephmeral groups");
  for (auto i : util::permute(s.nentities(), rng)) {
    const size_t gid = s.remove_value(i, rng);

    // delete all empty groups
    // [except if we created an empty group by calling remove_value()]
    bool match = false;
    for (auto g : s.empty_groups()) {
      if (g == gid) {
        // Alg 8 allows us to keep the value drawn from G_0 in this case:
        // "If c_i \neq c_j for all j \neq i, let c_i have the label k^- + 1,
        // and draw values independently from G_0 for those \phi_c for which
        // k^- + 1 < c \leq h."
        match = true;
        continue;
      }
      s.delete_group(g);
    }
    MICROSCOPES_ASSERT(s.empty_groups().size() == 0 ||
                       s.empty_groups().size() == 1);

    // create m new groups
    for (size_t g = (match ? 1 : 0); g < m; g++)
      s.create_group(rng);

    MICROSCOPES_ASSERT(s.empty_groups().size() == m);

    auto scores = s.score_value(i, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    s.add_value(choice, i, rng);
  }
}

void
gibbs::hp(fixed_entity_based_state_object &s,
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
      s.set_component_hp(fid, *g0.first);
      scores.push_back(g0.second + s.score_likelihood(fid, rng));
    }
    const auto choice = util::sample_discrete_log(scores, rng);
    s.set_component_hp(fid, *g[choice].first);
  }
}

// for performance debugging purposes
// doesn't change the group assignments
void
gibbs::perftest(entity_based_state_object &s, rng_t &rng)
{
  AssertAllAssigned(s);
  for (auto i : util::permute(s.nentities(), rng)) {
    const size_t gid = s.remove_value(i, rng);
    auto scores = s.score_value(i, rng);
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    (void)choice; // XXX: make sure compiler does not optimize this out
    s.add_value(gid, i, rng);
  }
}
