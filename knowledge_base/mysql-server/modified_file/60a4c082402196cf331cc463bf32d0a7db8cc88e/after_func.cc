    as Item_in_subselect::fix_after_pullout() will do this.
    So, just forward the call to the Item_in_subselect object.
  */

  args[1]->fix_after_pullout(parent_select, removed_select);

  used_tables_cache |= args[1]->used_tables();
  not_null_tables_cache |= args[1]->not_null_tables();
}

/**
   The implementation of optimized @<outer expression@> [NOT] IN @<subquery@>
   predicates. It applies to predicates which have gone through the IN->EXISTS
   transformation in in_to_exists_transformer functions; not to subquery
   materialization (which has no triggered conditions).

   The implementation works as follows.
   For the current value of the outer expression

   - If it contains only NULL values, the original (before rewrite by the
     Item_in_subselect rewrite methods) inner subquery is non-correlated and
     was previously executed, there is no need to re-execute it, and the
     previous return value is returned.

   - If it contains NULL values, check if there is a partial match for the
     inner query block by evaluating it. For clarity we repeat here the
     transformation previously performed on the sub-query. The expression

     <tt>
     ( oc_1, ..., oc_n )
     @<in predicate@>
     ( SELECT ic_1, ..., ic_n
       FROM @<table@>
       WHERE @<inner where@>
     )
     </tt>

     was transformed into

     <tt>
     ( oc_1, ..., oc_n )
     \@in predicate@>
     ( SELECT ic_1, ..., ic_n
       FROM @<table@>
       WHERE @<inner where@> AND ... ( ic_k = oc_k OR ic_k IS NULL )
       HAVING ... NOT ic_k IS NULL
     )
