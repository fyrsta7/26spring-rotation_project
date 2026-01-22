	DecompressChunkState *state = (DecompressChunkState *) node;
	CustomScan *cscan = castNode(CustomScan, node->ss.ps.plan);
	Plan *compressed_scan = linitial(cscan->custom_plans);
	Assert(list_length(cscan->custom_plans) == 1);

	if (eflags & EXEC_FLAG_BACKWARD)
		state->reverse = !state->reverse;

	state->hypertable_compression_info = get_hypertablecompression_info(state->hypertable_id);

	initialize_column_state(state);

	node->custom_ps = lappend(node->custom_ps, ExecInitNode(compressed_scan, estate, eflags));
}

static void
initialize_batch(DecompressChunkState *state, TupleTableSlot *slot)
{
	Datum value;
	bool isnull;
	int i;

	for (i = 0; i < state->num_columns; i++)
	{
		DecompressChunkColumnState *column = &state->columns[i];

		switch (column->type)
		{
			case COMPRESSED_COLUMN:
			{
				value = slot_getattr(slot, AttrOffsetGetAttrNumber(i), &isnull);
				if (!isnull)
				{
					CompressedDataHeader *header = (CompressedDataHeader *) PG_DETOAST_DATUM(value);

					column->compressed.iterator =
						tsl_get_decompression_iterator_init(header->compression_algorithm,
															state->reverse)(value, column->typid);
				}
				else
					column->compressed.iterator = NULL;

				break;
			}
			case SEGMENTBY_COLUMN:
				value = slot_getattr(slot, AttrOffsetGetAttrNumber(i), &isnull);
				if (!isnull)
					column->segmentby.value = value;
