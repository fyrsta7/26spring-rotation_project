			 * and not checking for prefix key equality yet, so we can't
			 * assume the group pivot tuple will reamin the same -- unless
			 * we're using a minimum group size of 1, in which case the pivot
			 * is obviously still the pviot.
			 */
			if (nTuples != minGroupSize)
				ExecClearTuple(node->group_pivot);
		}


		/*
		 * Pull as many tuples from the outer node as possible given our
		 * current operating mode.
		 */
		for (;;)
		{
			slot = ExecProcNode(outerNode);

			/*
			 * If the outer node can't provide us any more tuples, then we can
			 * sort the current group and return those tuples.
			 */
			if (TupIsNull(slot))
			{
				/*
				 * We need to know later if the outer node has completed to be
				 * able to distinguish between being done with a batch and
				 * being done with the whole node.
				 */
				node->outerNodeDone = true;

				SO1_printf("Sorting fullsort with %ld tuples\n", nTuples);
				tuplesort_performsort(fullsort_state);

				INSTRUMENT_SORT_GROUP(node, fullsort)

				SO_printf("Setting execution_status to INCSORT_READFULLSORT (final tuple)\n");
				node->execution_status = INCSORT_READFULLSORT;
				break;
