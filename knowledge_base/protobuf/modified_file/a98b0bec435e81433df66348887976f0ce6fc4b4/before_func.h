class PROTOBUF_EXPORT ImplicitWeakMessage final : public MessageLite {
 public:
  ImplicitWeakMessage() : ImplicitWeakMessage(nullptr) {}
  explicit constexpr ImplicitWeakMessage(ConstantInitialized);
  ImplicitWeakMessage(const ImplicitWeakMessage&) = delete;
  ImplicitWeakMessage& operator=(const ImplicitWeakMessage&) = delete;

  // Arena enabled constructors: for internal use only.
  ImplicitWeakMessage(internal::InternalVisibility, Arena* arena)
      : ImplicitWeakMessage(arena) {}

  // TODO: make this constructor private
  explicit ImplicitWeakMessage(Arena* arena)
      : MessageLite(arena, class_data_.base()), data_(new std::string) {}

  ~ImplicitWeakMessage() PROTOBUF_FINAL {
    // data_ will be null in the default instance, but we can safely call delete
    // here because the default instance will never be destroyed.
    delete data_;
  }

  static const ImplicitWeakMessage& default_instance();

  const ClassData* GetClassData() const PROTOBUF_FINAL;

  void Clear() PROTOBUF_FINAL { data_->clear(); }

  size_t ByteSizeLong() const PROTOBUF_FINAL {
    size_t size = data_ == nullptr ? 0 : data_->size();
    cached_size_.Set(internal::ToCachedSize(size));
    return size;
  }

  uint8_t* _InternalSerialize(
      uint8_t* target, io::EpsCopyOutputStream* stream) const PROTOBUF_FINAL {
    if (data_ == nullptr) {
      return target;
    }
    return stream->WriteRaw(data_->data(), static_cast<int>(data_->size()),
                            target);
  }

  typedef void InternalArenaConstructable_;

  static PROTOBUF_CC const char* ParseImpl(ImplicitWeakMessage* msg,
                                           const char* ptr, ParseContext* ctx);

 private:
  static const TcParseTable<0> table_;
  static const ClassDataLite<1> class_data_;

  static void MergeImpl(MessageLite&, const MessageLite&);

  static void DestroyImpl(MessageLite& msg) {
    static_cast<ImplicitWeakMessage&>(msg).~ImplicitWeakMessage();
  }
  static size_t ByteSizeLongImpl(const MessageLite& msg) {
    return static_cast<const ImplicitWeakMessage&>(msg).ByteSizeLong();
  }

  static uint8_t* _InternalSerializeImpl(const MessageLite& msg,
                                         uint8_t* target,
                                         io::EpsCopyOutputStream* stream) {
    return static_cast<const ImplicitWeakMessage&>(msg)._InternalSerialize(
        target, stream);
  }

  // This std::string is allocated on the heap, but we use a raw pointer so that
  // the default instance can be constant-initialized. In the const methods, we
  // have to handle the possibility of data_ being null.
  std::string* data_;
  google::protobuf::internal::CachedSize cached_size_{};
};
