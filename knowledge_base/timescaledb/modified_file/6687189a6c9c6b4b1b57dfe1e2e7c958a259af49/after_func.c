	Datum val;

	/* is this a compressed column or a segment-by column */
	bool is_compressed;

	/* the value stored in the compressed table was NULL */
	bool is_null;

	/* the index in the decompressed table of the data -1,
	 * if the data is metadata not found in the decompressed table
	 */
	int16 decompressed_column_offset;
} PerCompressedColumn;

typedef struct RowDecompressor
{
	PerCompressedColumn *per_compressed_cols;
	int16 num_compressed_columns;

	TupleDesc out_desc;
	Relation out_rel;

	CommandId mycid;
	BulkInsertState bistate;

	/* cache memory used to store the decompressed datums/is_null for form_tuple */
	Datum *decompressed_datums;
	bool *decompressed_is_nulls;
} RowDecompressor;

static PerCompressedColumn *create_per_compressed_column(TupleDesc in_desc, TupleDesc out_desc,
														 Oid out_relid,
														 Oid compressed_data_type_oid);
static void populate_per_compressed_columns_from_data(PerCompressedColumn *per_compressed_cols,
													  int16 num_cols, Datum *compressed_datums,
													  bool *compressed_is_nulls);
static void row_decompressor_decompress_row(RowDecompressor *row_decompressor);
static bool per_compressed_col_get_data(PerCompressedColumn *per_compressed_col,
										Datum *decompressed_datums, bool *decompressed_is_nulls);

void
decompress_chunk(Oid in_table, Oid out_table)
{
	/* these locks are taken in the order uncompressed table then compressed table
	 * for consistency with compress_chunk
	 */
	/* we are _just_ INSERTing into the out_table so in principle we could take
	 * a RowExclusive lock, and let other operations read and write this table
	 * as we work. However, we currently compress each table as a oneshot, so
	 * we're taking the stricter lock to prevent accidents.
	 */
	Relation out_rel = relation_open(out_table, ExclusiveLock);
	/*We want to prevent other decompressors from decompressing this table,
	 * and we want to prevent INSERTs or UPDATEs which could mess up our decompression.
	 * We may as well allow readers to keep reading the compressed data while
	 * we are compressing, so we only take an ExclusiveLock instead of AccessExclusive.
	 */
	Relation in_rel = relation_open(in_table, ExclusiveLock);
	// TODO error if out_rel is non-empty

	TupleDesc in_desc = RelationGetDescr(in_rel);
	TupleDesc out_desc = RelationGetDescr(out_rel);

	Oid compressed_data_type_oid = ts_custom_type_cache_get(CUSTOM_TYPE_COMPRESSED_DATA)->type_oid;

	Assert(in_desc->natts >= out_desc->natts);
	Assert(OidIsValid(compressed_data_type_oid));

	{
		RowDecompressor decompressor = {
			.per_compressed_cols = create_per_compressed_column(in_desc,
																out_desc,
																out_table,
																compressed_data_type_oid),
			.num_compressed_columns = in_desc->natts,

			.out_desc = out_desc,
			.out_rel = out_rel,
