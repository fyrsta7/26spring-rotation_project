
inline uint32_t grpc_slice_refcount::Hash(const grpc_slice& slice) {
  switch (ref_type_) {
    case Type::STATIC:
      return ::grpc_static_metadata_hash_values[GRPC_STATIC_METADATA_INDEX(
          slice)];
