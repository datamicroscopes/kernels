#include <microscopes/kernels/gibbs.hpp>
#include <microscopes/common/util.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::mixture;
using namespace microscopes::kernels;

void
gibbs::assign(state &s, dataview &view, rng_t &rng)
{
  // ensure 1 empty group
  // note: this is more efficient than s.ensure_k_empty_groups(1, rng)
  size_t egid = 0;
  const size_t egsizeinit = s.emptygroups().size();
  if (!egsizeinit)
    egid = s.create_group(rng);
  else {
    auto it = s.emptygroups().begin();
    egid = *it++;
    if (egsizeinit > 1) {
      vector<size_t> egremove(it, s.emptygroups().end());
      for (auto g : egremove)
        s.remove_group(g);
    }
  }

  //cout << "empty group: " << egid << endl;
  for (view.reset(); !view.end(); view.next()) {
    const size_t gid = s.remove_value(view, rng);
    //cout << "datapoint " << view.index() << " belongs to group " << gid << endl;
    if (!s.groupsize(gid)) {
      //cout << "  * remove from group " << gid << endl;
      s.remove_group(gid);
    }
    row_accessor acc = view.get();
    auto scores = s.score_value(acc, rng);
    //cout << "  * scores: " << scores.second << endl;
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    //cout << "  * probs: " << scores.second << endl;
    //cout << "  * add to group " << choice << endl;
    s.add_value(choice, view, rng);
    if (choice == egid) {
      egid = s.create_group(rng);
      //cout << "  * new empty group: " << egid << endl;
    }
  }
  //cout << "currently " << s.ngroups() << endl;
}
