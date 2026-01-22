bool MemTable::Get(const LookupKey& key, std::string* value, Status* s,
                  const Options& options) {
  Slice memkey = key.memtable_key();
  Table::Iterator iter(&table_);
  iter.Seek(memkey.data());

  bool merge_in_progress = false;
  std::string operand;
  if (s->IsMergeInProgress()) {
    swap(*value, operand);
    merge_in_progress = true;
  }


  auto merge_operator = options.merge_operator;
  auto logger = options.info_log;
  for (; iter.Valid(); iter.Next()) {
    // entry format is:
    //    klength  varint32
    //    userkey  char[klength-8]
    //    tag      uint64
    //    vlength  varint32
    //    value    char[vlength]
    // Check that it belongs to same user key.  We do not check the
    // sequence number since the Seek() call above should have skipped
    // all entries with overly large sequence numbers.
    const char* entry = iter.key();
    uint32_t key_length;
    const char* key_ptr = GetVarint32Ptr(entry, entry+5, &key_length);
    if (comparator_.comparator.user_comparator()->Compare(
            Slice(key_ptr, key_length - 8),
            key.user_key()) == 0) {
      // Correct user key
      const uint64_t tag = DecodeFixed64(key_ptr + key_length - 8);
      switch (static_cast<ValueType>(tag & 0xff)) {
        case kTypeValue: {
          Slice v = GetLengthPrefixedSlice(key_ptr + key_length);
          if (merge_in_progress) {
            merge_operator->Merge(key.user_key(), &v, operand,
                                   value, logger.get());
          } else {
            value->assign(v.data(), v.size());
          }
          return true;
        }
        case kTypeMerge: {
          Slice v = GetLengthPrefixedSlice(key_ptr + key_length);
          if (merge_in_progress) {
            merge_operator->Merge(key.user_key(), &v, operand,
                                  value, logger.get());
            swap(*value, operand);
          } else {
            assert(merge_operator);
            merge_in_progress = true;
            operand.assign(v.data(), v.size());
          }
          break;
        }
        case kTypeDeletion: {
          if (merge_in_progress) {
            merge_operator->Merge(key.user_key(), nullptr, operand,
                                   value, logger.get());
          } else {
            *s = Status::NotFound(Slice());
          }
          return true;
        }
      }
    } else {
      // exit loop if user key does not match
      break;
    }
  }

  if (merge_in_progress) {
    swap(*value, operand);
    *s = Status::MergeInProgress("");
  }
  return false;
}
