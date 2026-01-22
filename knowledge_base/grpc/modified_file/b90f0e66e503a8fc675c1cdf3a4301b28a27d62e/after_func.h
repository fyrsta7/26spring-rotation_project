    return false;
  }

  grpc::protobuf::int64 ByteCount() const override {
    return byte_count_ - backup_count_;
  }

 protected:
  int64_t byte_count_;
  int64_t backup_count_;
  grpc_byte_buffer_reader reader_;
  grpc_slice slice_;
  Status status_;
};

// BufferWriter must be a subclass of io::ZeroCopyOutputStream.
template <class BufferWriter, class T>
Status GenericSerialize(const grpc::protobuf::Message& msg,
                        grpc_byte_buffer** bp, bool* own_buffer) {
  static_assert(
      std::is_base_of<protobuf::io::ZeroCopyOutputStream, BufferWriter>::value,
      "BufferWriter must be a subclass of io::ZeroCopyOutputStream");
  *own_buffer = true;
