#include <microscopes/kernels/bootstrap.hpp>
#include <microscopes/common/util.hpp>
#include <microscopes/common/macros.hpp>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::mixture;
using namespace microscopes::kernels;

void
bootstrap::likelihood(state &s, dataview &view, rng_t &rng)
{
  MICROSCOPES_DCHECK(!s.ngroups(), "not a clean s");
  view.reset();
  s.add_value(s.create_group(rng), view, rng);
  size_t egid = s.create_group(rng);
  for (view.next(); !view.end(); view.next()) {
    row_accessor acc = view.get();
    auto scores = s.score_value(acc, rng);
    //cout << "  * scores: " << scores.second << endl;
    const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
    //cout << "  * probs: " << scores.second << endl;
    //cout << "  * add to group " << choice << endl;
    s.add_value(choice, view, rng);
    if (choice == egid)
      egid = s.create_group(rng);
  }
  //cout << "placed in " << s.ngroups() << endl;
}


