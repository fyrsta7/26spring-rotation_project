 */
static void warn_index_not_applicable(THD *thd, const RANGE_OPT_PARAM *param,
                                      const uint key_num, const Field *field) {
  if (param->using_real_indexes &&
      (thd->lex->is_explain() ||
       thd->variables.option_bits & OPTION_SAFE_UPDATES))
    push_warning_printf(thd, Sql_condition::SL_WARNING,
                        ER_WARN_INDEX_NOT_APPLICABLE,
                        ER_THD(thd, ER_WARN_INDEX_NOT_APPLICABLE), "range",
                        field->table->key_info[param->real_keynr[key_num]].name,
                        field->field_name);
}

/*
  Build a SEL_TREE for <> or NOT BETWEEN predicate

  SYNOPSIS
    get_ne_mm_tree()
      param       RANGE_OPT_PARAM from test_quick_select
      prev_tables See test_quick_select()
      read_tables See test_quick_select()
      remove_jump_scans See get_mm_tree()
      cond_func   item for the predicate
      field       field in the predicate
      lt_value    constant that field should be smaller
      gt_value    constant that field should be greaterr

  RETURN
    #  Pointer to tree built tree
    0  on error
*/
static SEL_TREE *get_ne_mm_tree(THD *thd, RANGE_OPT_PARAM *param,
                                table_map prev_tables, table_map read_tables,
                                bool remove_jump_scans, Item_func *cond_func,
                                Field *field, Item *lt_value, Item *gt_value) {
  SEL_TREE *tree = nullptr;

  if (param->has_errors()) return nullptr;

  tree = get_mm_parts(thd, param, prev_tables, read_tables, cond_func, field,
                      Item_func::LT_FUNC, lt_value);
  if (tree) {
    tree = tree_or(param, remove_jump_scans, tree,
                   get_mm_parts(thd, param, prev_tables, read_tables, cond_func,
                                field, Item_func::GT_FUNC, gt_value));
  }
  return tree;
}

/**
  Factory function to build a SEL_TREE from an @<in predicate@>

  @param thd        Thread handle
  @param param      Information on 'just about everything'.
  @param prev_tables See test_quick_select()
  @param read_tables See test_quick_select()
  @param remove_jump_scans See get_mm_tree()
  @param predicand  The @<in predicate's@> predicand, i.e. the left-hand
                    side of the @<in predicate@> expression.
  @param op         The 'in' operator itself.
  @param is_negated If true, the operator is NOT IN, otherwise IN.
*/
static SEL_TREE *get_func_mm_tree_from_in_predicate(
    THD *thd, RANGE_OPT_PARAM *param, table_map prev_tables,
    table_map read_tables, bool remove_jump_scans, Item *predicand,
    Item_func_in *op, bool is_negated) {
  if (param->has_errors()) return nullptr;

  // Populate array as we need to examine its values here
  if (op->m_const_array != nullptr && !op->m_populated) {
    op->populate_bisection(thd);
  }
  if (is_negated) {
    // We don't support row constructors (multiple columns on lhs) here.
    if (predicand->type() != Item::FIELD_ITEM) return nullptr;

    Field *field = down_cast<Item_field *>(predicand)->field;

    if (op->m_const_array != nullptr && !op->m_const_array->is_row_result()) {
      /*
        We get here for conditions on the form "t.key NOT IN (c1, c2, ...)",
        where c{i} are constants. Our goal is to produce a SEL_TREE that
        represents intervals:

        ($MIN<t.key<c1) OR (c1<t.key<c2) OR (c2<t.key<c3) OR ...    (*)

        where $MIN is either "-inf" or NULL.

        The most straightforward way to produce it is to convert NOT
        IN into "(t.key != c1) AND (t.key != c2) AND ... " and let the
        range analyzer build a SEL_TREE from that. The problem is that
        the range analyzer will use O(N^2) memory (which is probably a
        bug), and people who do use big NOT IN lists (e.g. see
        BUG#15872, BUG#21282), will run out of memory.

        Another problem with big lists like (*) is that a big list is
        unlikely to produce a good "range" access, while considering
        that range access will require expensive CPU calculations (and
        for MyISAM even index accesses). In short, big NOT IN lists
        are rarely worth analyzing.

        Considering the above, we'll handle NOT IN as follows:

        - if the number of entries in the NOT IN list is less than
          NOT_IN_IGNORE_THRESHOLD, construct the SEL_TREE (*)
          manually.

        - Otherwise, don't produce a SEL_TREE.
      */

      const uint NOT_IN_IGNORE_THRESHOLD = 1000;
      // If we have t.key NOT IN (null, null, ...) or the list is too long
      if (op->m_const_array->m_used_size == 0 ||
          op->m_const_array->m_used_size > NOT_IN_IGNORE_THRESHOLD)
        return nullptr;

      /*
        Create one Item_type constant object. We'll need it as
        get_mm_parts only accepts constant values wrapped in Item_Type
        objects.
        We create the Item on thd->mem_root which points to
        per-statement mem_root.
      */
      Item_basic_constant *value_item =
          op->m_const_array->create_item(thd->mem_root);
      if (value_item == nullptr) return nullptr;

      /* Get a SEL_TREE for "(-inf|NULL) < X < c_0" interval.  */
      uint i = 0;
      SEL_TREE *tree = nullptr;
      do {
        op->m_const_array->value_to_item(i, value_item);
        tree = get_mm_parts(thd, param, prev_tables, read_tables, op, field,
                            Item_func::LT_FUNC, value_item);
        if (!tree) break;
        i++;
      } while (i < op->m_const_array->m_used_size &&
               tree->type == SEL_TREE::IMPOSSIBLE);

      if (!tree || tree->type == SEL_TREE::IMPOSSIBLE)
        /* We get here in cases like "t.unsigned NOT IN (-1,-2,-3) */
        return nullptr;
      SEL_TREE *tree2 = nullptr;
      Item_basic_constant *previous_range_value =
          op->m_const_array->create_item(thd->mem_root);
      for (; i < op->m_const_array->m_used_size; i++) {
        // Check if the value stored in the field for the previous range
        // is greater, lesser or equal to the actual value specified in the
        // query. Used further down to set the flags for the current range
        // correctly (as the max value for the previous range will become
        // the min value for the current range).
        op->m_const_array->value_to_item(i - 1, previous_range_value);
        int cmp_value =
            stored_field_cmp_to_item(thd, field, previous_range_value);
        if (op->m_const_array->compare_elems(i, i - 1)) {
          /* Get a SEL_TREE for "-inf < X < c_i" interval */
          op->m_const_array->value_to_item(i, value_item);
          tree2 = get_mm_parts(thd, param, prev_tables, read_tables, op, field,
                               Item_func::LT_FUNC, value_item);
          if (!tree2) {
            tree = nullptr;
            break;
          }

          /* Change all intervals to be "c_{i-1} < X < c_i" */
          for (uint idx = 0; idx < param->keys; idx++) {
            SEL_ARG *last_val;
            if (tree->keys[idx] && tree2->keys[idx] &&
                ((last_val = tree->keys[idx]->root->last()))) {
              SEL_ARG *new_interval = tree2->keys[idx]->root;
              new_interval->min_value = last_val->max_value;
              // We set the max value of the previous range as the beginning
              // for this range interval. However we need values higher than
              // this value:
              // For ex: If the range is "not in (1,2)" we first construct
              // X < 1 before this loop and add 1 < X < 2 in this loop and
              // follow it up with 2 < X below.
              // While fetching values for the second interval, we set
              // "NEAR_MIN" flag so that we fetch values higher than "1".
              // However, when the values specified are not compatible
              // with the field that is being compared to, they are rounded
              // off.
              // For the example above, if the range given was "not in (0.9,
              // 1.9)", range optimizer rounds of the values to (1,2). In such
              // a case, setting the flag to "NEAR_MIN" is not right. Because
              // we need values higher than "0.9" not "1". We check this
              // before we set the flag below.
              if (cmp_value <= 0)
                new_interval->min_flag = NEAR_MIN;
              else
