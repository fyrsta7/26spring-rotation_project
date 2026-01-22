
static uint
cache_record_length(JOIN *join,uint idx)
{
  uint length=0;
  JOIN_TAB **pos,**end;
  THD *thd=join->thd;

  for (pos=join->best_ref+join->const_tables,end=join->best_ref+idx ;
       pos != end ;
       pos++)
  {
    JOIN_TAB *join_tab= *pos;
    if (!join_tab->used_fieldlength)		/* Not calced yet */
      calc_used_field_length(thd, join_tab);
    length+=join_tab->used_fieldlength;
  }
  return length;
}


/**
  Find the best index to do 'ref' access on for a table.

  @param tab                        the table to be joined by the function
  @param remaining_tables           set of tables not included in the
                                    partial plan yet.
  @param idx                        the index in join->position[] where 'tab'
                                    is added to the partial plan.
  @param prefix_rowcount            estimate for the number of records returned
                                    by the partial plan
  @param found_condition [out]      whether or not there exists a condition
                                    that filters away rows for this table.
                                    Always true when the function finds a
                                    usable 'ref' access, but also if it finds
                                    a condition that is not usable by 'ref'
                                    access, e.g. is there is an index covering
                                    (a,b) and there is a condition only on 'b'.
                                    Note that all dependent tables for the
                                    condition in question must be in the plan
                                    prefix for this to be 'true'. Unmodified
                                    if no relevant condition is found.
  @param ref_depend_map [out]       tables the best ref access depends on.
                                    Unmodified if no 'ref' access is found.
  @param used_key_parts [out]       Number of keyparts 'ref' access uses.
                                    Unmodified if no 'ref' access is found.

  @return pointer to Key_use for the index with best 'ref' access, NULL if
          no 'ref' access method is found.
*/
Key_use* Optimize_table_order::find_best_ref(const JOIN_TAB *tab,
                                             const table_map remaining_tables,
                                             const uint idx,
                                             const double prefix_rowcount,
                                             bool *found_condition,
                                             table_map *ref_depend_map,
                                             uint *used_key_parts)
{
  // Return value - will point to Key_use of the index with cheapest ref access
  Key_use *best_ref= NULL;

  /*
    Cost of using best_ref; used to determine if ref access on another
    index is cheaper. Calculated as follows:

    (cost_ref_for_one_value + ROW_EVALUATE_COST * fanout_for_ref) *
    prefix_rowcount
  */
  double best_ref_cost= DBL_MAX;

  TABLE *const table= tab->table;
  Opt_trace_context *const trace= &thd->opt_trace;

  /*
    Guessing the number of distinct values in the table; used to
    make "rec_per_key"-like estimates when no statistics is
    available.
  */
  ha_rows distinct_keys_est= tab->records/MATCHING_ROWS_IN_OTHER_TABLE;

  // Test how we can use keys
  for (Key_use *keyuse= tab->keyuse; keyuse->table == table; )
  {
    // keyparts that are usable for this index given the current partial plan
    key_part_map found_part= 0;
    // Bitmap of keyparts where the ref access is over 'keypart=const'
    key_part_map const_part= 0;
    /*
      Cost of ref access on current index. Calculated as follows:
      cost_ref_for_one_value * prefix_rowcount
    */
    double cur_read_cost;
    // Fanout for ref access using this index
    double cur_fanout;
    uint cur_used_keyparts= 0;  // number of used keyparts
    // tables 'ref' access on this index depends on
    table_map table_deps= 0;
    const uint key= keyuse->key;
    const KEY *const keyinfo= table->key_info + key;
    const bool ft_key= (keyuse->keypart == FT_KEYPART);
    /*
      Bitmap of keyparts in this index that have a condition 

        "WHERE col=... OR col IS NULL"

      If 'ref' access is to be used in such cases, the JT_REF_OR_NULL
      type will be used.
    */
    key_part_map ref_or_null_part= 0;

    DBUG_PRINT("info", ("Considering ref access on key %s", keyinfo->name));
    Opt_trace_object trace_access_idx(trace);
    trace_access_idx.add_alnum("access_type", "ref").
      add_utf8("index", keyinfo->name);

    // Calculate how many key segments of the current key we can use
    Key_use *const start_key= keyuse;
    start_key->bound_keyparts= 0;  // Initially, no ref access is possible

    // For each keypart
    while (keyuse->table == table && keyuse->key == key)
    {
      const uint keypart= keyuse->keypart;
      // tables the current keypart depends on
      table_map cur_keypart_table_deps= 0;
      double best_distinct_prefix_rowcount= DBL_MAX;

      /*
        Check all ways to access the keypart. There is one keyuse
        object for each equality predicate for the keypart, and this
        loop estimates which equality predicate is best. Example that
        would have two keyuse objects for a keypart covering
        t1.col_x: "WHERE t1.col_x=4 AND t1.col_x=t2.col_y"
      */
      for ( ; keyuse->table == table && keyuse->key == key &&
              keyuse->keypart == keypart ; ++keyuse)
      {
        /*
          This keyuse cannot be used if 
          1) it is a key reference between a table inside a semijoin
             nest and one outside of it. The same applices to
             materialized subqueries
          2) it is a key reference to a table that is not in the plan
             prefix (i.e., a table that will be later in the join
             sequence)
          3) there will be two ref_or_null keyparts 
             ("WHERE col=... OR col IS NULL"). Thus if
             a) the condition for an earlier keypart is of type
                ref_or_null, and
             b) the condition for the current keypart is ref_or_null
        */
        if ((excluded_tables & keyuse->used_tables) ||        // 1)
            (remaining_tables & keyuse->used_tables) ||       // 2)
            (ref_or_null_part &&                              // 3a)
             (keyuse->optimize & KEY_OPTIMIZE_REF_OR_NULL)))  // 3b)
          continue;

        found_part|= keyuse->keypart_map;
        if (!(keyuse->used_tables & ~join->const_table_map))
          const_part|= keyuse->keypart_map;

        const double cur_distinct_prefix_rowcount=
          prev_record_reads(join, idx, (table_deps | keyuse->used_tables));
        if (cur_distinct_prefix_rowcount < best_distinct_prefix_rowcount)
        {
          /*
            We estimate that the currently considered usage of the
            keypart will have to lookup fewer distinct key
            combinations from the prefix tables.
          */
          cur_keypart_table_deps= keyuse->used_tables & ~join->const_table_map;
          best_distinct_prefix_rowcount= cur_distinct_prefix_rowcount;
        }
        if (distinct_keys_est > keyuse->ref_table_rows)
          distinct_keys_est= keyuse->ref_table_rows;
        /*
          If there is one 'key_column IS NULL' expression, we can
          use this ref_or_null optimisation of this field
        */
        if (keyuse->optimize & KEY_OPTIMIZE_REF_OR_NULL)
          ref_or_null_part|= keyuse->keypart_map;
      }
      table_deps|= cur_keypart_table_deps;
    }

    if (distinct_keys_est < MATCHING_ROWS_IN_OTHER_TABLE)
      // Fix for small tables
      distinct_keys_est= MATCHING_ROWS_IN_OTHER_TABLE;

    // fulltext indexes require special treatment
    if (!ft_key)
    {
      *found_condition|= test(found_part);

      // Check if we found full key
      if (found_part == LOWER_BITS(key_part_map,
                                   actual_key_parts(keyinfo)) &&
          !ref_or_null_part)
      {                                         /* use eq key */
        cur_used_keyparts= (uint) ~0;
        if ((keyinfo->flags & (HA_NOSAME | HA_NULL_PART_KEY)) == HA_NOSAME)
        {
          cur_read_cost= prev_record_reads(join, idx, table_deps);
          cur_fanout= 1.0;
        }
        else
        {
          if (!table_deps)
          {                                     /* We found a const key */
            /*
              ReuseRangeEstimateForRef-1:
              We get here if we've found a ref(const) (c_i are constants):
              "(keypart1=c1) AND ... AND (keypartN=cN)"   [ref_const_cond]

              If range optimizer was able to construct a "range"
              access on this index, then its condition "quick_cond" was
              eqivalent to ref_const_cond (*), and we can re-use E(#rows)
              from the range optimizer.

              Proof of (*): By properties of range and ref optimizers
              quick_cond will be equal or tighter than ref_const_cond.
              ref_const_cond already covers "smallest" possible interval -
              a singlepoint interval over all keyparts. Therefore,
              quick_cond is equivalent to ref_const_cond (if it was an
              empty interval we wouldn't have got here).
            */
            if (table->quick_keys.is_set(key))
              cur_fanout= (double) table->quick_rows[key];
            else
            {
              // quick_range couldn't use key
              cur_fanout= (double) tab->records/distinct_keys_est;
            }
          }
          else
          {
            // Use rec_per_key statistics if available
            if (keyinfo->rec_per_key[actual_key_parts(keyinfo)-1])
              cur_fanout= keyinfo->rec_per_key[actual_key_parts(keyinfo)-1];
            else
            {                              /* Prefer longer keys */
              cur_fanout=
                ((double) tab->records / (double) distinct_keys_est *
                 (1.0 +
                  ((double) (table->s->max_key_length-keyinfo->key_length) /
                   (double) table->s->max_key_length)));
              if (cur_fanout < 2.0)
                cur_fanout= 2.0;        /* Can't be as good as a unique */
            }

            /*
              ReuseRangeEstimateForRef-2:  We get here if we could not reuse
              E(#rows) from range optimizer. Make another try:

              If range optimizer produced E(#rows) for a prefix of the ref
              access we're considering, and that E(#rows) is lower then our
              current estimate, make an adjustment. The criteria of when we
              can make an adjustment is a special case of the criteria used
              in ReuseRangeEstimateForRef-3.
            */
            if (table->quick_keys.is_set(key) &&
                (const_part &
                 (((key_part_map)1 << table->quick_key_parts[key])-1)) ==
                (((key_part_map)1 << table->quick_key_parts[key])-1) &&
                table->quick_n_ranges[key] == 1 &&
                cur_fanout > (double) table->quick_rows[key])
            {
              cur_fanout= (double) table->quick_rows[key];
            }
          }
          // Limit the number of matched rows
          const double tmp_fanout=
            min(cur_fanout, (double) thd->variables.max_seeks_for_key);
          if (table->covering_keys.is_set(key))
          {
            // We can use only index tree
            cur_read_cost=
              prefix_rowcount *
              table->file->index_only_read_time(key, tmp_fanout);
          }
          else
            cur_read_cost= prefix_rowcount * min(tmp_fanout, tab->worst_seeks);
        }
      }
      else if ((found_part & 1) &&
               (!(table->file->index_flags(key, 0, 0) & HA_ONLY_WHOLE_INDEX) ||
                found_part == LOWER_BITS(key_part_map,
