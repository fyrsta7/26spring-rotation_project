		 * support the convention that sk_subtype == InvalidOid means the
		 * opclass input type; this is a hack to simplify life for
		 * ScanKeyInit().
		 */
		elemtype = cur->sk_subtype;
		if (elemtype == InvalidOid)
			elemtype = rel->rd_opcintype[cur->sk_attno - 1];
		Assert(elemtype == ARR_ELEMTYPE(arrayval));

		/*
		 * If the comparison operator is not equality, then the array qual
		 * degenerates to a simple comparison against the smallest or largest
		 * non-null array element, as appropriate.
		 */
		switch (cur->sk_strategy)
		{
			case BTLessStrategyNumber:
			case BTLessEqualStrategyNumber:
				cur->sk_argument =
					_bt_find_extreme_element(scan, cur, elemtype,
											 BTGreaterStrategyNumber,
											 elem_values, num_nonnulls);
				continue;
			case BTEqualStrategyNumber:
				/* proceed with rest of loop */
				break;
			case BTGreaterEqualStrategyNumber:
			case BTGreaterStrategyNumber:
				cur->sk_argument =
					_bt_find_extreme_element(scan, cur, elemtype,
											 BTLessStrategyNumber,
											 elem_values, num_nonnulls);
				continue;
			default:
				elog(ERROR, "unrecognized StrategyNumber: %d",
					 (int) cur->sk_strategy);
				break;
		}

		/*
		 * We'll need a 3-way ORDER proc to perform binary searches for the
		 * next matching array element.  Set that up now.
		 *
		 * Array scan keys with cross-type equality operators will require a
		 * separate same-type ORDER proc for sorting their array.  Otherwise,
		 * sortproc just points to the same proc used during binary searches.
		 */
		_bt_setup_array_cmp(scan, cur, elemtype,
							&so->orderProcs[i], &sortprocp);

		/*
		 * Sort the non-null elements and eliminate any duplicates.  We must
		 * sort in the same ordering used by the index column, so that the
		 * arrays can be advanced in lockstep with the scan's progress through
		 * the index's key space.
		 */
		reverse = (indoption[cur->sk_attno - 1] & INDOPTION_DESC) != 0;
		num_elems = _bt_sort_array_elements(cur, sortprocp, reverse,
											elem_values, num_nonnulls);

		if (origarrayatt == cur->sk_attno)
		{
			BTArrayKeyInfo *orig = &so->arrayKeys[origarraykey];

			/*
			 * This array scan key is redundant with a previous equality
			 * operator array scan key.  Merge the two arrays together to
			 * eliminate contradictory non-intersecting elements (or try to).
			 *
			 * We merge this next array back into attribute's original array.
			 */
			Assert(arrayKeyData[orig->scan_key].sk_attno == cur->sk_attno);
			Assert(arrayKeyData[orig->scan_key].sk_collation ==
				   cur->sk_collation);
			if (_bt_merge_arrays(scan, cur, sortprocp, reverse,
								 origelemtype, elemtype,
								 orig->elem_values, &orig->num_elems,
								 elem_values, num_elems))
			{
				/* Successfully eliminated this array */
				pfree(elem_values);

				/*
				 * If no intersecting elements remain in the original array,
				 * the scan qual is unsatisfiable
				 */
				if (orig->num_elems == 0)
				{
					so->qual_ok = false;
					break;
				}

				/*
				 * Indicate to _bt_preprocess_keys caller that it must ignore
				 * this scan key
				 */
				cur->sk_strategy = InvalidStrategy;
				continue;
			}

			/*
			 * Unable to merge this array with previous array due to a lack of
			 * suitable cross-type opfamily support.  Will need to keep both
			 * scan keys/arrays.
