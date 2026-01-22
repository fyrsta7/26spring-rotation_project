        grpc_slice_from_copied_string(key1);
      metadata->metadata[metadata->count].value =
        grpc_slice_from_copied_buffer(Z_STRVAL_P(value), Z_STRLEN_P(value));
      metadata->count += 1;
    PHP_GRPC_HASH_FOREACH_END()
  PHP_GRPC_HASH_FOREACH_END()
  return true;
}

void grpc_php_metadata_array_destroy_including_entries(
