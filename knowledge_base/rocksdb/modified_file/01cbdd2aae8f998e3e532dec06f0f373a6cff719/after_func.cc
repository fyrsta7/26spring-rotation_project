Options GetRocksDBOptionsFromOptions(const SpatialDBOptions& options) {
  Options rocksdb_options;
  rocksdb_options.IncreaseParallelism(options.num_threads);
  rocksdb_options.write_buffer_size = 256 * 1024 * 1024;          // 256MB
  rocksdb_options.max_bytes_for_level_base = 1024 * 1024 * 1024;  // 1 GB
  // only compress levels >= 1
  rocksdb_options.compression_per_level.resize(rocksdb_options.num_levels);
  for (int i = 0; i < rocksdb_options.num_levels; ++i) {
    if (i == 0) {
      rocksdb_options.compression_per_level[i] = kNoCompression;
    } else {
      rocksdb_options.compression_per_level[i] = kLZ4Compression;
    }
  }
  rocksdb_options.block_cache = NewLRUCache(options.cache_size);
  if (options.bulk_load) {
    rocksdb_options.PrepareForBulkLoad();
  }
  return rocksdb_options;
}
