    }

    DEBUG_PRINTF("accel %s+%u\n", describeClass(ei.cr).c_str(), ei.offset);

    const CharReach &escape = ei.cr;
    auto nonexit_symbols = find_nonexit_symbols(rdfa, escape);

    vector<dstate_id_t> pending = {base};
    while (!pending.empty()) {
        dstate_id_t curr = pending.back();
        pending.pop_back();
        for (auto s : nonexit_symbols) {
            dstate_id_t t = rdfa.states[curr].next[s];
            if (contains(region, t)) {
                continue;
            }

            DEBUG_PRINTF("    %hu is in region\n", t);
            region.insert(t);
            pending.push_back(t);
        }
    }

    return region;
}

AccelScheme
accel_dfa_build_strat::find_escape_strings(dstate_id_t this_idx) const {
    AccelScheme rv;
    const raw_dfa &rdfa = get_raw();
    rv.cr.clear();
    rv.offset = 0;
    const dstate &raw = rdfa.states[this_idx];
    const vector<CharReach> rev_map = reverse_alpha_remapping(rdfa);
    bool outs2_broken = false;
    map<dstate_id_t, CharReach> succs;

    for (u32 i = 0; i < rev_map.size(); i++) {
        if (raw.next[i] == this_idx) {
            continue;
        }

        const CharReach &cr_i = rev_map.at(i);

        rv.cr |= cr_i;
        dstate_id_t next_id = raw.next[i];

        DEBUG_PRINTF("next is %hu\n", next_id);
        const dstate &raw_next = rdfa.states[next_id];

        if (outs2_broken) {
            continue;
        }

        if (!raw_next.reports.empty() && generates_callbacks(rdfa.kind)) {
            DEBUG_PRINTF("leads to report\n");
            outs2_broken = true; /* cannot accelerate over reports */
            continue;
        }
        succs[next_id] |= cr_i;
    }

    if (!outs2_broken) {
        for (const auto &e : succs) {
            const CharReach &cr_i = e.second;
            const dstate &raw_next = rdfa.states[e.first];

            CharReach cr_all_j;
            for (u32 j = 0; j < rev_map.size(); j++) {
                if (raw_next.next[j] == raw.next[j]) {
                    continue;
                }

                DEBUG_PRINTF("state %hu: adding sym %u -> %hu to 2 \n", e.first,
                             j, raw_next.next[j]);
                cr_all_j |= rev_map.at(j);
            }

            if (cr_i.count() * cr_all_j.count() > 8) {
                DEBUG_PRINTF("adding %zu to double_cr\n", cr_i.count());
                rv.double_cr |= cr_i;
            } else {
                for (auto ii = cr_i.find_first(); ii != CharReach::npos;
                     ii = cr_i.find_next(ii)) {
                    for (auto jj = cr_all_j.find_first(); jj != CharReach::npos;
                         jj = cr_all_j.find_next(jj)) {
                        rv.double_byte.emplace((u8)ii, (u8)jj);
                        if (rv.double_byte.size() > 8) {
                            DEBUG_PRINTF("outs2 too big\n");
                            outs2_broken = true;
                            goto done;
                        }
                    }
                }
