Options GetRocksDBOptionsFromOptions(const SpatialDBOptions& options) {
  Options rocksdb_options;
  rocksdb_options.OptimizeLevelStyleCompaction();
  rocksdb_options.IncreaseParallelism(options.num_threads);
  rocksdb_options.block_cache = NewLRUCache(options.cache_size);
  if (options.bulk_load) {
    rocksdb_options.PrepareForBulkLoad();
  }
  return rocksdb_options;
}
