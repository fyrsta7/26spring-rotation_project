template <class Type>
inline PROTOBUF_ALWAYS_INLINE const char* ParseVarint(const char* p,
                                                      Type* value) {
  static_assert(sizeof(Type) == 4 || sizeof(Type) == 8,
                "Only [u]int32_t and [u]int64_t please");
  int64_t byte = static_cast<int8_t>(*p);
  if (PROTOBUF_PREDICT_TRUE(byte >= 0)) {
    *value = byte;
    return p + 1;
  } else {
    auto tmp = Parse64FallbackPair(p, byte);
    if (PROTOBUF_PREDICT_TRUE(tmp.first)) {
      *value = static_cast<Type>(tmp.second);
    }
    return tmp.first;
  }
}
