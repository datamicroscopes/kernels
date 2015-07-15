#pragma once

#include <microscopes/common/random_fwd.hpp>
#include <microscopes/common/entity_state.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>

#include <vector>
#include <utility>

namespace microscopes {
namespace kernels {

struct gibbs {
    typedef std::vector<std::pair<const models::hypers *, float>> grid_t;

    static void
    assign(common::entity_based_state_object &state, common::rng_t &rng);

    static void
    assign_resample(common::entity_based_state_object &state, size_t m, common::rng_t &rng);

    static void
    hp(common::entity_based_state_object &state,
       const std::vector<std::pair<size_t, grid_t>> &params,
       common::rng_t &rng);

    static void
    perftest(common::entity_based_state_object &state,
             common::rng_t &rng);
};

// --- templated kernel interface which each project will specialize ---
// --- XXX(stephentu): transition existing codebase to use gibbsT

struct gibbsT {
    template <typename state_t> static void assign(state_t &t, common::rng_t &rng);
    template <typename state_t> static void assign2(state_t &t, common::rng_t &rng);
};

// --- implementation of gibbsT ---

namespace detail {

static inline ALWAYS_INLINE void
AssertAllAssigned(const std::vector<ssize_t> &assignments)
{
#ifdef DEBUG_MODE
    for (auto i : assignments)
        MICROSCOPES_DCHECK(i != -1, "unassigned entity found");
#endif /* DEBUG_MODE */
}

static inline ALWAYS_INLINE void
AssertAllAssigned2(const std::vector<std::vector<ssize_t>> &assignments)
{
#ifdef DEBUG_MODE
    for (const auto &a : assignments)
        for (auto i : a)
            MICROSCOPES_DCHECK(i != -1, "unassigned entity found");
#endif /* DEBUG_MODE */
}

} // namespace detail

//
//  Concept class:
//
//  class state_t {
//    std::vector<ssize_t> assignments() const;
//    size_t nentities() const;
//
//    std::set<size_t> empty_groups() const;
//    size_t create_group(common::rng_t &);
//    void delete_group(size_t gid);
//    size_t groupsize(size_t gid) const;
//
//    size_t remove_value(size_t eid, common::rng_t &);
//    void inplace_score_value(
//      std::pair<std::vector<size_t>, std::vector<float>> &,
//      size_t eid,
//      common::rng_t &);
//    void add_value(size_t eid, size_t gid, common::rng_t &);
//
//  };
//
template <typename state_t>
void
gibbsT::assign(state_t &state, common::rng_t &rng)
{
    detail::AssertAllAssigned(state.assignments());
    std::pair<std::vector<size_t>, std::vector<float>> scores;
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
    for (auto i : common::util::permute(state.nentities(), rng)) {
        const size_t gid = state.remove_value(i, rng);
        if (!state.groupsize(gid))
            state.delete_group(gid);
        MICROSCOPES_ASSERT(state.empty_groups().size() == 1);
        state.inplace_score_value(scores, i, rng);
        const auto choice = scores.first[common::util::sample_discrete_log(scores.second, rng)];
        state.add_value(i, choice, rng);
        if (choice == egid)
            egid = state.create_group(rng);
    }
}

//
//  Concept class:
//
//  class state_t {
//    std::vector<std::vector<ssize_t>> assignments() const;
//    size_t nentities() const;
//    size_t nterms(size_t) const;
//
//    std::set<size_t> empty_dishes() const;
//    size_t create_dish(common::rng_t &);
//    void delete_dish(size_t);
//    size_t dishsize(size_t) const;
//
//    std::set<size_t> empty_tables(size_t) const;
//    size_t create_table(size_t, common::rng_t &);
//    void delete_table(size_t, size_t);
//    size_t tablesize(size_t, size_t) const;
//
//    std::pair<size_t, size_t> remove_value(size_t, size_t, common::rng_t &);
//    void inplace_score_value(
//      std::pair<std::vector<size_t>, std::vector<float>> &,
//      size_t,
//      size_t,
//      common::rng_t &);
//    size_t add_value(size_t, size_t, size_t, common::rng_t &);
//
//  };
//
template <typename state_t>
void
gibbsT::assign2(state_t &state, common::rng_t &rng)
{
    detail::AssertAllAssigned2(state.assignments());
    std::pair<std::vector<size_t>, std::vector<float>> scores;

    // ensure 1 empty dish
    const auto empty_dishes = state.empty_dishes();
    size_t edid = 0;
    if (empty_dishes.empty()) {
        edid = state.create_dish(rng);
    } else {
        auto it = empty_dishes.begin();
        edid = *it++;
        for (; it != empty_dishes.end(); ++it)
            state.delete_dish(*it);
    }

    for (auto i : common::util::permute(state.nentities(), rng)) {
        // ensure 1 empty table
        const auto empty_tables = state.empty_tables(i);
        size_t etid = 0;
        if (empty_tables.empty()) {
            etid = state.create_table(i, rng);
        } else {
            auto it = empty_tables.begin();
            etid = *it++;
            for (; it != empty_tables.end(); ++it)
                state.delete_table(i, *it);
        }

        for (auto j : common::util::permute(state.nterms(i), rng)) {
            const auto p = state.remove_value(i, j, rng);
            const size_t did = p.first;
            const size_t tid = p.second;
            if (!state.tablesize(i, tid))
                state.delete_table(i, tid);
            if (!state.dishsize(did))
                state.delete_dish(did);
            MICROSCOPES_ASSERT(state.empty_dishes().size() == 1);
            MICROSCOPES_ASSERT(state.empty_tables(i).size() == 1);
            state.inplace_score_value(scores, i, j, rng);
            const auto new_tid = scores.first[common::util::sample_discrete_log(scores.second, rng)];
            const size_t new_did = state.add_value(i, j, new_tid, rng);
            if (new_did == edid)
                edid = state.create_dish(rng);
            if (new_tid == etid)
                etid = state.create_table(i, rng);
        }
    }
}

} // namespace kernels
} // namespace microscopes
