void MessageGenerator::GenerateSerializeWithCachedSizesBody(io::Printer* p) {
  if (HasSimpleBaseClass(descriptor_, options_)) return;
  Formatter format(p);
  // If there are multiple fields in a row from the same oneof then we
  // coalesce them and emit a switch statement.  This is more efficient
  // because it lets the C++ compiler know this is a "at most one can happen"
  // situation. If we emitted "if (has_x()) ...; if (has_y()) ..." the C++
  // compiler's emitted code might check has_y() even when has_x() is true.
  class LazySerializerEmitter {
   public:
    LazySerializerEmitter(MessageGenerator* mg, io::Printer* p)
        : mg_(mg),
          p_(p),
          eager_(mg->descriptor_->file()->syntax() ==
                 FileDescriptor::SYNTAX_PROTO3),
          cached_has_bit_index_(kNoHasbit) {}

    ~LazySerializerEmitter() { Flush(); }

    // If conditions allow, try to accumulate a run of fields from the same
    // oneof, and handle them at the next Flush().
    void Emit(const FieldDescriptor* field) {
      Formatter format(p_);
      if (eager_ || MustFlush(field)) {
        Flush();
      }
      if (!field->real_containing_oneof()) {
        // TODO(ckennelly): Defer non-oneof fields similarly to oneof fields.

        if (!field->options().weak() && !field->is_repeated() && !eager_) {
          // We speculatively load the entire _has_bits_[index] contents, even
          // if it is for only one field.  Deferring non-oneof emitting would
          // allow us to determine whether this is going to be useful.
          int has_bit_index = mg_->has_bit_indices_[field->index()];
          if (cached_has_bit_index_ != has_bit_index / 32) {
            // Reload.
            int new_index = has_bit_index / 32;

            format("cached_has_bits = _impl_._has_bits_[$1$];\n", new_index);

            cached_has_bit_index_ = new_index;
          }
        }

        mg_->GenerateSerializeOneField(p_, field, cached_has_bit_index_);
      } else {
        v_.push_back(field);
      }
    }

    void EmitIfNotNull(const FieldDescriptor* field) {
      if (field != nullptr) {
        Emit(field);
      }
    }

    void Flush() {
      if (!v_.empty()) {
        mg_->GenerateSerializeOneofFields(p_, v_);
        v_.clear();
      }
    }

   private:
    // If we have multiple fields in v_ then they all must be from the same
    // oneof.  Would adding field to v_ break that invariant?
    bool MustFlush(const FieldDescriptor* field) {
      return !v_.empty() &&
             v_[0]->containing_oneof() != field->containing_oneof();
    }

    MessageGenerator* mg_;
    io::Printer* p_;
    bool eager_;
    std::vector<const FieldDescriptor*> v_;

    // cached_has_bit_index_ maintains that:
    //   cached_has_bits = from._has_bits_[cached_has_bit_index_]
    // for cached_has_bit_index_ >= 0
    int cached_has_bit_index_;
  };

  class LazyExtensionRangeEmitter {
   public:
    LazyExtensionRangeEmitter(MessageGenerator* mg, io::Printer* p)
        : mg_(mg), p_(p) {}

    void AddToRange(const Descriptor::ExtensionRange* range) {
      if (!has_current_range_) {
        current_combined_range_ = *range;
        has_current_range_ = true;
      } else {
        current_combined_range_.start =
            std::min(current_combined_range_.start, range->start);
        current_combined_range_.end =
            std::max(current_combined_range_.end, range->end);
      }
    }

    void Flush() {
      if (has_current_range_) {
        mg_->GenerateSerializeOneExtensionRange(p_, &current_combined_range_);
      }
      has_current_range_ = false;
    }

   private:
    MessageGenerator* mg_;
    io::Printer* p_;
    bool has_current_range_ = false;
    Descriptor::ExtensionRange current_combined_range_;
  };

  // We need to track the largest weak field, because weak fields are serialized
  // differently than normal fields.  The WeakFieldMap::FieldWriter will
  // serialize all weak fields that are ordinally between the last serialized
  // weak field and the current field.  In order to guarantee that all weak
  // fields are serialized, we need to make sure to emit the code to serialize
  // the largest weak field present at some point.
  class LargestWeakFieldHolder {
   public:
    const FieldDescriptor* Release() {
      const FieldDescriptor* result = field_;
      field_ = nullptr;
      return result;
    }
    void ReplaceIfLarger(const FieldDescriptor* field) {
      if (field_ == nullptr || field_->number() < field->number()) {
        field_ = field;
      }
    }

   private:
    const FieldDescriptor* field_ = nullptr;
  };

  std::vector<const FieldDescriptor*> ordered_fields =
      SortFieldsByNumber(descriptor_);

  std::vector<const Descriptor::ExtensionRange*> sorted_extensions;
  sorted_extensions.reserve(descriptor_->extension_range_count());
  for (int i = 0; i < descriptor_->extension_range_count(); ++i) {
    sorted_extensions.push_back(descriptor_->extension_range(i));
  }
  std::sort(sorted_extensions.begin(), sorted_extensions.end(),
            ExtensionRangeSorter());
  if (num_weak_fields_) {
    format(
        "::_pbi::WeakFieldMap::FieldWriter field_writer("
        "$weak_field_map$);\n");
  }

  format(
      "$uint32$ cached_has_bits = 0;\n"
      "(void) cached_has_bits;\n\n");

  // Merge the fields and the extension ranges, both sorted by field number.
  {
    LazySerializerEmitter e(this, p);
    LazyExtensionRangeEmitter re(this, p);
    LargestWeakFieldHolder largest_weak_field;
    int i, j;
    for (i = 0, j = 0;
         i < ordered_fields.size() || j < sorted_extensions.size();) {
      if ((j == sorted_extensions.size()) ||
          (i < descriptor_->field_count() &&
           ordered_fields[i]->number() < sorted_extensions[j]->start)) {
        const FieldDescriptor* field = ordered_fields[i++];
        re.Flush();
        if (field->options().weak()) {
          largest_weak_field.ReplaceIfLarger(field);
          PrintFieldComment(format, field);
        } else {
          e.EmitIfNotNull(largest_weak_field.Release());
          e.Emit(field);
        }
      } else {
        e.EmitIfNotNull(largest_weak_field.Release());
        e.Flush();
        re.AddToRange(sorted_extensions[j++]);
      }
    }
    re.Flush();
    e.EmitIfNotNull(largest_weak_field.Release());
  }

  format("if (PROTOBUF_PREDICT_FALSE($have_unknown_fields$)) {\n");
  format.Indent();
  if (UseUnknownFieldSet(descriptor_->file(), options_)) {
    format(
        "target = "
        "::_pbi::WireFormat::"
        "InternalSerializeUnknownFieldsToArray(\n"
        "    $unknown_fields$, target, stream);\n");
  } else {
    format(
        "target = stream->WriteRaw($unknown_fields$.data(),\n"
        "    static_cast<int>($unknown_fields$.size()), target);\n");
  }
  format.Outdent();
  format("}\n");
}
