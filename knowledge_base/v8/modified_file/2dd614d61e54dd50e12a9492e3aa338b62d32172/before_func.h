template <typename ValidationTag, typename... Args>
void DecodeError(Decoder* decoder, const byte* pc, const char* str,
                 Args&&... args) {
  if constexpr (!ValidationTag::validate) UNREACHABLE();
  static_assert(sizeof...(Args) > 0);
  if constexpr (ValidationTag::full_validation) {
    decoder->errorf(pc, str, std::forward<Args>(args)...);
  } else {
    decoder->MarkError();
  }
}
