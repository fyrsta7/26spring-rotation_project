    if (window->frame_buffer() == nullptr) {
      CreateFramebufferTable(thd, *path->window().temp_table_param,
                             *query_block, original_fields, *join->fields,
                             temp_table_param->items_to_copy, window);
    }
  } else {
    for (Func_ptr &func : *temp_table_param->items_to_copy) {
      // Even without buffering, some window functions will read
      // their arguments out of the output table, so we need to apply
      // our own temporary table to them. (For cases with buffering,
      // this replacement, or a less strict version, is done in
      // CreateFramebufferTable().)
      if (func.should_copy(CFT_HAS_WF) || func.should_copy(CFT_WF)) {
        ReplaceMaterializedItems(thd, func.func(),
                                 *temp_table_param->items_to_copy,
                                 /*need_exact_match=*/true);
      }
    }
  }
  window->make_special_rows_cache(thd, path->window().temp_table);
}

static bool ContainsMaterialization(AccessPath *path) {
  bool found = false;
  WalkAccessPaths(path, /*join=*/nullptr,
                  WalkAccessPathPolicy::STOP_AT_MATERIALIZATION,
                  [&found](AccessPath *sub_path, JOIN *) {
                    if (sub_path->type == AccessPath::MATERIALIZE) {
                      found = true;
                    }
                    return found;
                  });
  return found;
}

static Item *AddCachesAroundConstantConditions(Item *item) {
  cache_const_expr_arg cache_arg;
  cache_const_expr_arg *analyzer_arg = &cache_arg;
  return item->compile(
      &Item::cache_const_expr_analyzer, pointer_cast<uchar **>(&analyzer_arg),
      &Item::cache_const_expr_transformer, pointer_cast<uchar *>(&cache_arg));
}

[[nodiscard]] static bool AddCachesAroundConstantConditionsInPath(
    AccessPath *path) {
  // TODO(sgunders): We could probably also add on sort and GROUP BY
  // expressions, even though most of them should have been removed by the
  // interesting order framework. The same with the SELECT list and
  // expressions used in materializations.
  switch (path->type) {
    case AccessPath::FILTER:
      path->filter().condition =
          AddCachesAroundConstantConditions(path->filter().condition);
      return path->filter().condition == nullptr;
    case AccessPath::HASH_JOIN:
      for (Item *&item :
           path->hash_join().join_predicate->expr->join_conditions) {
        item = AddCachesAroundConstantConditions(item);
        if (item == nullptr) {
          return true;
        }
      }
      return false;
    default:
      return false;
  }
}

/*
  Do the final touchups of the access path tree, once we have selected a final
  plan (ie., there are no more alternatives). There are currently two major
  tasks to do here: Account for materializations (because we cannot do it until
  we have the entire plan), and set up filesorts (because it involves
  constructing new objects, so we don't want to do it for unused candidates).
  The former also influences the latter.

  Materializations in particular are a bit tricky due to the way our item system
  works; expression evaluation cares intimately about _where_ values come from,
  not just what they are (i.e., all non-leaf Items carry references to other
  Items, and pull data only from there). Thus, whenever an Item is materialized,
  references to that Item need to be modified to instead point into the correct
  field in the temporary table. We traverse the tree bottom-up and keep track of
  which materializations are active, and modify the appropriate Item lists at
  any given point, so that they point to the right place. We currently modify:

    - The SELECT list. (There is only one, so we can update it as we go.)
    - Referenced fields for INSERT ... ON DUPLICATE KEY UPDATE (IODKU);
      also updated as we go.
    - Sort keys (e.g. for ORDER BY).
    - The HAVING clause, if the materialize node is below an aggregate node.
      (If the materialization is above aggregation, the HAVING clause has
      already accomplished its mission of filtering out the uninteresting
      results, and will not be evaluated anymore.)

  Surprisingly enough, we also need to update the materialization parameters
  themselves. Say that we first have a materialization that copies
  t1.x -> <temp1>.x. After that, we have a materialization that copies
  t1.x -> <temp2>.x. For this to work properly, we obviously need to go in
  and modify the second one so that it instead says <temp1>.x -> <temp2>.x,
  ie., the copy is done from the correct source.

  You cannot yet insert temporary tables in arbitrary places in the query;
  in particular, we do not yet handle these rewrites (although they would
  very likely be possible):

    - Group elements for aggregations (GROUP BY). Do note that
      create_tmp_table() will replace elements within aggregate functions
      if you set save_sum_funcs=false; you may also want to supplant
      this mechanism.
    - Filters (e.g. WHERE predicates); do note that partial pushdown may
      present its own challenges.
    - Join conditions.
 */
bool FinalizePlanForQueryBlock(THD *thd, Query_block *query_block) {
  assert(query_block->join->needs_finalize);
  query_block->join->needs_finalize = false;

  AccessPath *const root_path = query_block->join->root_access_path();
  assert(root_path != nullptr);
  if (root_path->type == AccessPath::EQ_REF) {
    // None of the finalization below is relevant to point selects, so just
    // return immediately.
    return false;
  }

  // If the query is offloaded to an external executor, we don't need to create
  // the internal temporary tables or filesort objects, or rewrite the Item tree
  // to point into them.
  if (!IteratorsAreNeeded(thd, root_path)) {
    return false;
  }

  Query_block *old_query_block = thd->lex->current_query_block();
  thd->lex->set_current_query_block(query_block);

  // We might have stacked multiple FILTERs on top of each other.
  // Combine these into a single FILTER:
  WalkAccessPaths(
      root_path, query_block->join, WalkAccessPathPolicy::ENTIRE_QUERY_BLOCK,
      [](AccessPath *path, JOIN *join [[maybe_unused]]) {
        if (path->type == AccessPath::FILTER) {
          AccessPath *child = path->filter().child;
