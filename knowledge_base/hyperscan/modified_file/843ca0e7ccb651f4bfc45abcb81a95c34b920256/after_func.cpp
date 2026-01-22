                .color_map(make_iterator_property_map(
                    colour.begin(), get(&NFAGraphVertexProps::index, g.g))));
    } catch (fas_visitor *) {
        ; /* found max accel_states */
    }

    remove_edges(topEdges, g);

    assert(out.size() <= NFA_MAX_ACCEL_STATES);
    accel_map->swap(out);
}

static
bool containsBadSubset(const limex_accel_info &accel,
                       const NFAStateSet &state_set, const u32 effective_sds) {
    NFAStateSet subset(state_set.size());
    for (size_t j = state_set.find_first(); j != state_set.npos;
         j = state_set.find_next(j)) {
        subset = state_set;
        subset.reset(j);

        if (effective_sds != NO_STATE && subset.count() == 1 &&
            subset.test(effective_sds)) {
            continue;
        }

        if (subset.any() && !contains(accel.precalc, subset)) {
            return true;
        }
    }
    return false;
}

static
void doAccelCommon(NGHolder &g,
                   ue2::unordered_map<NFAVertex, AccelScheme> &accel_map,
                   const ue2::unordered_map<NFAVertex, u32> &state_ids,
                   const map<NFAVertex, BoundedRepeatSummary> &br_cyclic,
                   const u32 num_states, limex_accel_info *accel,
                   const CompileContext &cc) {
    vector<CharReach> refined_cr = reduced_cr(g, br_cyclic);

    vector<NFAVertex> astates;
    for (const auto &m : accel_map) {
        astates.push_back(m.first);
    }

    NFAStateSet useful(num_states);
    NFAStateSet state_set(num_states);
    vector<NFAVertex> states;

    NFAVertex sds_or_proxy = get_sds_or_proxy(g);
    const u32 effective_sds = state_ids.at(sds_or_proxy);

    /* for each subset of the accel keys need to find an accel scheme */
    assert(astates.size() < 32);
    sort(astates.begin(), astates.end(), make_index_ordering(g));

    for (u32 i = 1, i_end = 1U << astates.size(); i < i_end; i++) {
        DEBUG_PRINTF("saving info for accel %u\n", i);
        states.clear();
        state_set.reset();
        for (u32 j = 0, j_end = astates.size(); j < j_end; j++) {
            if (i & (1U << j)) {
                NFAVertex v = astates[j];
                states.push_back(v);
                state_set.set(state_ids.at(v));
            }
        }

        if (containsBadSubset(*accel, state_set, effective_sds)) {
            DEBUG_PRINTF("accel %u has bad subset\n", i);
            continue; /* if a subset failed to build we would too */
        }

        const bool allow_wide = allow_wide_accel(states, g, sds_or_proxy);

        AccelScheme as = nfaFindAccel(g, states, refined_cr, br_cyclic,
                                      allow_wide);
        if (as.cr.count() > MAX_MERGED_ACCEL_STOPS) {
            DEBUG_PRINTF("accel %u too wide (%zu, %d)\n", i,
                         as.cr.count(), MAX_MERGED_ACCEL_STOPS);
            continue;
        }

        DEBUG_PRINTF("accel %u ok with offset %u\n", i, as.offset);

        // try multibyte acceleration first
        MultibyteAccelInfo mai = nfaCheckMultiAccel(g, states, cc);

        precalcAccel &pa = accel->precalc[state_set];
        useful |= state_set;

        // if we successfully built a multibyte accel scheme, use that
        if (mai.type != MultibyteAccelInfo::MAT_NONE) {
            pa.ma_info = mai;

            DEBUG_PRINTF("multibyte acceleration!\n");
            continue;
        }

        pa.single_offset = as.offset;
        pa.single_cr = as.cr;

        if (states.size() == 1) {
            DoubleAccelInfo b = findBestDoubleAccelInfo(g, states.front());
            if (pa.single_cr.count() > b.stop1.count()) {
                /* insert this information into the precalc accel info as it is
                 * better than the single scheme */
