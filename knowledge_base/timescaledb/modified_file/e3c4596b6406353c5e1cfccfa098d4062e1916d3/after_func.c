		table_close(rel, NoLock);
	}

#if PG14_LT
	int options = 0;
#else
	ReindexParams params = { 0 };
	ReindexParams *options = &params;
#endif
	reindex_relation(table_oid, REINDEX_REL_PROCESS_TOAST, options);
	rel = table_open(table_oid, AccessExclusiveLock);
	CommandCounterIncrement();
	table_close(rel, NoLock);
}

CompressionStats
compress_chunk(Oid in_table, Oid out_table, const ColumnCompressionInfo **column_compression_info,
			   int num_compression_infos, int insert_options)
{
	int n_keys;
	ListCell *lc;
	int indexscan_direction = NoMovementScanDirection;
	Relation matched_index_rel = NULL;
	TupleTableSlot *slot;
	IndexScanDesc index_scan;
	CommandId mycid = GetCurrentCommandId(true);
	HeapTuple in_table_tp = NULL, index_tp = NULL;
	Form_pg_attribute in_table_attr_tp, index_attr_tp;
	const ColumnCompressionInfo **keys;
	CompressionStats cstat;

	/* We want to prevent other compressors from compressing this table,
	 * and we want to prevent INSERTs or UPDATEs which could mess up our compression.
	 * We may as well allow readers to keep reading the uncompressed data while
	 * we are compressing, so we only take an ExclusiveLock instead of AccessExclusive.
	 */
	Relation in_rel = table_open(in_table, ExclusiveLock);
	/* We are _just_ INSERTing into the out_table so in principle we could take
	 * a RowExclusive lock, and let other operations read and write this table
	 * as we work. However, we currently compress each table as a oneshot, so
	 * we're taking the stricter lock to prevent accidents.
	 */
	Relation out_rel = relation_open(out_table, ExclusiveLock);
	int16 *in_column_offsets = compress_chunk_populate_keys(in_table,
															column_compression_info,
															num_compression_infos,
															&n_keys,
															&keys);

	TupleDesc in_desc = RelationGetDescr(in_rel);
	TupleDesc out_desc = RelationGetDescr(out_rel);
	/* Before calling row compressor relation should be segmented and sorted as per
	 * compress_segmentby and compress_orderby column/s configured in ColumnCompressionInfo.
	 * Cost of sorting can be mitigated if we find an existing BTREE index defined for
	 * uncompressed chunk otherwise expensive tuplesort will come into play.
	 *
	 * The following code is trying to find an existing index that
	 * matches the ColumnCompressionInfo so that we can skip sequential scan and
	 * tuplesort.
	 *
	 * Matching Criteria for Each IndexAtt[i] and ColumnCompressionInfo Keys[i]
	 * ========================================================================
	 * a) Index attnum must match with ColumnCompressionInfo Key {keys[i]}.
	 * b) Index attOption(ASC/DESC and NULL_FIRST) can be mapped with ColumnCompressionInfo
	 * orderby_asc and null_first.
	 *
	 * BTREE Indexes Ordering
	 * =====================
	 * a) ASC[Null_Last] ==> [1]->[2]->NULL
	 * b) [Null_First]ASC ==> NULL->[1]->[2]
	 * c) DSC[Null_Last]  ==> [2]->[1]->NULL
	 * d) [Null_First]DSC ==> NULL->[2]->[1]
	 */
	if (ts_guc_enable_compression_indexscan)
	{
		List *in_rel_index_oids = RelationGetIndexList(in_rel);
		foreach (lc, in_rel_index_oids)
		{
			Oid index_oid = lfirst_oid(lc);
			Relation index_rel = index_open(index_oid, AccessShareLock);
			IndexInfo *index_info = BuildIndexInfo(index_rel);

			if (index_info->ii_Predicate != 0)
			{
				/*
				 * Can't use partial indexes for compression because they refer
				 * only to a subset of all rows.
				 */
				index_close(index_rel, AccessShareLock);
				continue;
			}

			int previous_direction = NoMovementScanDirection;
			int current_direction = NoMovementScanDirection;

			if (n_keys <= index_info->ii_NumIndexKeyAttrs && index_info->ii_Am == BTREE_AM_OID)
			{
				int i;
				for (i = 0; i < n_keys; i++)
				{
					int16 att_num = get_attnum(in_table, NameStr(keys[i]->attname));

					int16 option = index_rel->rd_indoption[i];
					bool index_orderby_asc = ((option & INDOPTION_DESC) == 0);
					bool index_null_first = ((option & INDOPTION_NULLS_FIRST) != 0);
					bool is_orderby_asc =
						COMPRESSIONCOL_IS_SEGMENT_BY(keys[i]) ? true : keys[i]->orderby_asc;
					bool is_null_first =
						COMPRESSIONCOL_IS_SEGMENT_BY(keys[i]) ? false : keys[i]->orderby_nullsfirst;

					if (att_num == 0 || index_info->ii_IndexAttrNumbers[i] != att_num)
					{
						break;
					}

					in_table_tp = SearchSysCacheAttNum(in_table, att_num);
					if (!HeapTupleIsValid(in_table_tp))
						elog(ERROR,
							 "table \"%s\" does not have column \"%s\"",
							 get_rel_name(in_table),
							 NameStr(keys[i]->attname));

					index_tp = SearchSysCacheAttNum(index_oid, i + 1);
					if (!HeapTupleIsValid(index_tp))
						elog(ERROR,
							 "index \"%s\" does not have column \"%s\"",
							 get_rel_name(index_oid),
							 NameStr(keys[i]->attname));

					in_table_attr_tp = (Form_pg_attribute) GETSTRUCT(in_table_tp);
					index_attr_tp = (Form_pg_attribute) GETSTRUCT(index_tp);

					if (index_orderby_asc == is_orderby_asc && index_null_first == is_null_first &&
						in_table_attr_tp->attcollation == index_attr_tp->attcollation)
					{
						current_direction = ForwardScanDirection;
					}
					else if (index_orderby_asc != is_orderby_asc &&
							 index_null_first != is_null_first &&
							 in_table_attr_tp->attcollation == index_attr_tp->attcollation)
					{
						current_direction = BackwardScanDirection;
					}
					else
					{
						current_direction = NoMovementScanDirection;
						break;
					}

					ReleaseSysCache(in_table_tp);
					in_table_tp = NULL;
					ReleaseSysCache(index_tp);
					index_tp = NULL;
					if (previous_direction == NoMovementScanDirection)
					{
						previous_direction = current_direction;
					}
					else if (previous_direction != current_direction)
					{
						break;
					}
				}

				if (n_keys == i && (previous_direction == current_direction &&
									current_direction != NoMovementScanDirection))
				{
					matched_index_rel = index_rel;
					indexscan_direction = current_direction;
					break;
				}
				else
				{
					if (HeapTupleIsValid(in_table_tp))
					{
						ReleaseSysCache(in_table_tp);
						in_table_tp = NULL;
					}
					if (HeapTupleIsValid(index_tp))
					{
						ReleaseSysCache(index_tp);
						index_tp = NULL;
					}
					index_close(index_rel, AccessShareLock);
				}
			}
			else
			{
				index_close(index_rel, AccessShareLock);
			}
		}
	}

	Assert(num_compression_infos <= in_desc->natts);
	Assert(num_compression_infos <= out_desc->natts);
	RowCompressor row_compressor;
	row_compressor_init(&row_compressor,
						in_desc,
						out_rel,
						num_compression_infos,
						column_compression_info,
						in_column_offsets,
						out_desc->natts,
						true /*need_bistate*/,
						false /*reset_sequence*/,
						insert_options);

	if (matched_index_rel != NULL)
	{
		if (ts_guc_debug_compression_path_info)
		{
			elog(INFO,
				 "compress_chunk_indexscan_start matched index \"%s\"",
