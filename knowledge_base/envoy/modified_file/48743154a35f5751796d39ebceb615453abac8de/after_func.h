
  // Config::GrpcMuxCallbacks
  void onConfigUpdate(const Protobuf::RepeatedPtrField<ProtobufWkt::Any>& resources,
                      const std::string& version_info) override {
    Protobuf::RepeatedPtrField<ResourceType> typed_resources;
    std::transform(resources.cbegin(), resources.cend(),
                   Protobuf::RepeatedPtrFieldBackInserter(&typed_resources),
                   MessageUtil::anyConvert<ResourceType>);
    callbacks_->onConfigUpdate(typed_resources);
    stats_.update_success_.inc();
    stats_.update_attempt_.inc();
    version_info_ = version_info;
    stats_.version_.set(HashUtil::xxHash64(version_info_));
    if (ENVOY_LOG_CHECK_LEVEL(debug)) {
      ENVOY_LOG(debug, "gRPC config for {} accepted with {} resources: {}", type_url_,
                resources.size(), RepeatedPtrUtil::debugString(typed_resources));
