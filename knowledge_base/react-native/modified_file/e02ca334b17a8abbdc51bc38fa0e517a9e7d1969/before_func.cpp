void RawPropsParser::postPrepare() noexcept {
  ready_ = true;
  nameToIndex_.reindex();
}

void RawPropsParser::preparse(const RawProps& rawProps) const noexcept {
  const size_t keyCount = keys_.size();
  rawProps.keyIndexToValueIndex_.resize(keyCount, kRawPropsValueIndexEmpty);

  // Resetting the cursor, the next increment will give `0`.
  rawProps.keyIndexCursor_ = static_cast<int>(keyCount - 1);

  // If the Props constructor doesn't use ::at at all, we might be
  // able to skip this entirely (in those cases, the Props struct probably
  // uses setProp instead).
  if (keyCount == 0) {
    return;
  }

  switch (rawProps.mode_) {
    case RawProps::Mode::Empty:
      return;

    case RawProps::Mode::JSI: {
      auto& runtime = *rawProps.runtime_;
      if (!rawProps.value_.isObject()) {
        LOG(ERROR) << "Preparse props: rawProps value is not object";
      }
      react_native_assert(rawProps.value_.isObject());
      auto object = rawProps.value_.asObject(runtime);

      auto names = object.getPropertyNames(runtime);
      auto count = names.size(runtime);
      auto valueIndex = RawPropsValueIndex{0};

      for (size_t i = 0; i < count; i++) {
        auto nameValue = names.getValueAtIndex(runtime, i).getString(runtime);
        auto value = object.getProperty(runtime, nameValue);

        auto name = nameValue.utf8(runtime);

        auto keyIndex = nameToIndex_.at(
            name.data(), static_cast<RawPropsPropNameLength>(name.size()));

        if (keyIndex == kRawPropsValueIndexEmpty) {
          continue;
        }

        rawProps.keyIndexToValueIndex_[keyIndex] = valueIndex;
        rawProps.values_.push_back(
            RawValue(jsi::dynamicFromValue(runtime, value)));
        valueIndex++;
      }

      break;
    }

    case RawProps::Mode::Dynamic: {
      const auto& dynamic = rawProps.dynamic_;
      auto valueIndex = RawPropsValueIndex{0};

      for (const auto& pair : dynamic.items()) {
        auto name = pair.first.getString();

        auto keyIndex = nameToIndex_.at(
            name.data(), static_cast<RawPropsPropNameLength>(name.size()));

        if (keyIndex == kRawPropsValueIndexEmpty) {
          continue;
        }

        rawProps.keyIndexToValueIndex_[keyIndex] = valueIndex;
        rawProps.values_.push_back(RawValue{pair.second});
        valueIndex++;
