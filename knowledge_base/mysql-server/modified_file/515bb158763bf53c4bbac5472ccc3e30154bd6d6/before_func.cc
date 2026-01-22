      thd->mem_root->ArrayAlloc<OrderElement>(m_longest_ordering);
  for (size_t state_idx = 0; state_idx < m_states.size(); ++state_idx) {
    // Refuse to apply FDs for nondeterministic orderings other than possibly
    // ordering -> grouping; ie., (a) can _not_ be satisfied by (a, rand()).
    // This is to avoid evaluating such a nondeterministic function unexpectedly
    // early, e.g. in GROUP BY when the user didn't expect it to be used in
    // ORDER BY. (We still allow it on exact matches, though.See also comments
    // on RAND_TABLE_BIT in SortAheadOrdering.)
    Ordering old_ordering = m_states[state_idx].satisfied_ordering;
    const bool deterministic = none_of(
        old_ordering.begin(), old_ordering.end(), [this](OrderElement element) {
          return Overlaps(m_items[element.item].item->used_tables(),
                          RAND_TABLE_BIT);
        });

    // Apply the special decay FD; first to convert it into a grouping
    // (which we always allow, even for nondeterministic items),
    // then to shorten the ordering.
    if (!IsGrouping(old_ordering) && !old_ordering.empty()) {
      AddGroupingFromOrdering(thd, state_idx, old_ordering, tmpbuf);
    }
    if (!deterministic) {
      continue;
    }
    if (!IsGrouping(old_ordering) && old_ordering.size() > 1) {
      AddEdge(thd, state_idx, /*required_fd_idx=*/0,
              old_ordering.without_back());
    }

    if (m_states.size() >= kMaxNFSMStates) {
      // Stop expanding new functional dependencies, causing us to end fairly
      // soon. We won't necessarily find the optimal query, but we'll keep all
      // essential information, and not throw away any of the information we
      // have already gathered (unless the DFSM gets too large, too;
      // see ConvertNFSMToDFSM()).
      continue;
    }

    for (size_t fd_idx = 1; fd_idx < m_fds.size(); ++fd_idx) {
      const FunctionalDependency &fd = m_fds[fd_idx];

      int start_point;
      if (!FunctionalDependencyApplies(fd, old_ordering, &start_point)) {
        continue;
      }

      ItemHandle item_to_add = fd.tail;

      // On a = b, try to replace a with b or b with a.
      Ordering base_ordering;
      if (fd.type == FunctionalDependency::EQUIVALENCE) {
        Ordering new_ordering{tmpbuf, old_ordering.size()};
        memcpy(tmpbuf, &old_ordering[0],
               sizeof(old_ordering[0]) * old_ordering.size());
        ItemHandle other_item = fd.head[0];
        if (new_ordering[start_point].item == item_to_add) {
          // b already existed, so it's a we must add.
          swap(item_to_add, other_item);
        }
        new_ordering[start_point].item = item_to_add;  // Keep the direction.
        DeduplicateOrdering(&new_ordering);
        if (CouldBecomeInterestingOrdering(new_ordering)) {
          AddEdge(thd, state_idx, fd_idx, new_ordering);
        }

        // Now we can add back the item we just replaced,
        // at any point after this. E.g., if we had an order abc
        // and applied b=d to get adc, we can add back b to get
        // adbc or adcb. Also, we'll fall through afterwards
        // to _not_ replacing but just adding d, e.g. abdc and abcd.
        // So fall through.
        base_ordering = new_ordering;
        item_to_add = other_item;
      } else {
        base_ordering = old_ordering;
      }

      // On S -> b, try to add b everywhere after the last element of S.
      if (IsGrouping(base_ordering)) {
        if (m_items[m_items[item_to_add].canonical_item].used_in_grouping) {
          TryAddingOrderWithElementInserted(
              thd, state_idx, fd_idx, base_ordering, /*start_point=*/0,
              item_to_add, ORDER_NOT_RELEVANT, tmpbuf2);
        }
      } else {
        // NOTE: We could have neither add_asc nor add_desc, if the item is used
        // only in groupings. If so, we don't add it at all, before we convert
        // it to a grouping.
        bool add_asc = m_items[m_items[item_to_add].canonical_item].used_asc;
        bool add_desc = m_items[m_items[item_to_add].canonical_item].used_desc;
        if (add_asc) {
          TryAddingOrderWithElementInserted(thd, state_idx, fd_idx,
                                            base_ordering, start_point + 1,
                                            item_to_add, ORDER_ASC, tmpbuf2);
        }
        if (add_desc) {
          TryAddingOrderWithElementInserted(thd, state_idx, fd_idx,
                                            base_ordering, start_point + 1,
                                            item_to_add, ORDER_DESC, tmpbuf2);
        }
      }
    }
  }
}

void LogicalOrderings::AddGroupingFromOrdering(THD *thd, int state_idx,
                                               Ordering ordering,
                                               OrderElement *tmpbuf) {
  memcpy(tmpbuf, &ordering[0], sizeof(*tmpbuf) * ordering.size());
  for (size_t i = 0; i < ordering.size(); ++i) {
    tmpbuf[i].direction = ORDER_NOT_RELEVANT;
    if (!m_items[m_items[tmpbuf[i].item].canonical_item].used_in_grouping) {
      // Pruned away.
      return;
    }
  }
  sort(tmpbuf, tmpbuf + ordering.size(),
       [this](const OrderElement &a, const OrderElement &b) {
         return ItemBeforeInGroup(a, b);
       });
  AddEdge(thd, state_idx, /*required_fd_idx=*/0,
          Ordering(tmpbuf, ordering.size()));
}

void LogicalOrderings::TryAddingOrderWithElementInserted(
    THD *thd, int state_idx, int fd_idx, Ordering old_ordering,
    size_t start_point, ItemHandle item_to_add, enum_order direction,
    OrderElement *tmpbuf) {
  if (static_cast<int>(old_ordering.size()) >= m_longest_ordering) {
