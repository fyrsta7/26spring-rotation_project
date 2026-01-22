void StringBuiltinsAssembler::StringIndexOf(
    Node* const subject_string, Node* const subject_instance_type,
    Node* const search_string, Node* const search_instance_type,
    Node* const position, std::function<void(Node*)> f_return) {
  CSA_ASSERT(this, IsString(subject_string));
  CSA_ASSERT(this, IsString(search_string));
  CSA_ASSERT(this, TaggedIsSmi(position));

  Node* const int_zero = IntPtrConstant(0);

  VARIABLE(var_needle_byte, MachineType::PointerRepresentation(), int_zero);
  VARIABLE(var_string_addr, MachineType::PointerRepresentation(), int_zero);

  Node* const search_length = SmiUntag(LoadStringLength(search_string));
  Node* const subject_length = SmiUntag(LoadStringLength(subject_string));
  Node* const start_position = IntPtrMax(SmiUntag(position), int_zero);

  Label zero_length_needle(this), return_minus_1(this);
  {
    GotoIf(IntPtrEqual(int_zero, search_length), &zero_length_needle);

    // Check that the needle fits in the start position.
    GotoIfNot(IntPtrLessThanOrEqual(search_length,
                                    IntPtrSub(subject_length, start_position)),
              &return_minus_1);
  }

  // Try to unpack subject and search strings. Bail to runtime if either needs
  // to be flattened.
  ToDirectStringAssembler subject_to_direct(state(), subject_string);
  ToDirectStringAssembler search_to_direct(state(), search_string);

  Label call_runtime_unchecked(this, Label::kDeferred);

  subject_to_direct.TryToDirect(&call_runtime_unchecked);
  search_to_direct.TryToDirect(&call_runtime_unchecked);

  // Load pointers to string data.
  Node* const subject_ptr =
      subject_to_direct.PointerToData(&call_runtime_unchecked);
  Node* const search_ptr =
      search_to_direct.PointerToData(&call_runtime_unchecked);

  Node* const subject_offset = subject_to_direct.offset();
  Node* const search_offset = search_to_direct.offset();

  // Like String::IndexOf, the actual matching is done by the optimized
  // SearchString method in string-search.h. Dispatch based on string instance
  // types, then call straight into C++ for matching.

  CSA_ASSERT(this, IntPtrGreaterThan(search_length, int_zero));
  CSA_ASSERT(this, IntPtrGreaterThanOrEqual(start_position, int_zero));
  CSA_ASSERT(this, IntPtrGreaterThanOrEqual(subject_length, start_position));
  CSA_ASSERT(this,
             IntPtrLessThanOrEqual(search_length,
                                   IntPtrSub(subject_length, start_position)));

  Label one_one(this), one_two(this), two_one(this), two_two(this);
  DispatchOnStringEncodings(subject_to_direct.instance_type(),
                            search_to_direct.instance_type(), &one_one,
                            &one_two, &two_one, &two_two);

  typedef const uint8_t onebyte_t;
  typedef const uc16 twobyte_t;

  BIND(&one_one);
  {
    Node* const adjusted_subject_ptr = PointerToStringDataAtIndex(
        subject_ptr, subject_offset, String::ONE_BYTE_ENCODING);
    Node* const adjusted_search_ptr = PointerToStringDataAtIndex(
        search_ptr, search_offset, String::ONE_BYTE_ENCODING);

    Label direct_memchr_call(this), generic_fast_path(this);
    Branch(IntPtrEqual(search_length, IntPtrConstant(1)), &direct_memchr_call,
           &generic_fast_path);

    // An additional fast path that calls directly into memchr for 1-length
    // search strings.
    BIND(&direct_memchr_call);
    {
      Node* const string_addr = IntPtrAdd(adjusted_subject_ptr, start_position);
      Node* const search_length = IntPtrSub(subject_length, start_position);
      Node* const search_byte =
          ChangeInt32ToIntPtr(Load(MachineType::Uint8(), adjusted_search_ptr));

      Node* const memchr =
          ExternalConstant(ExternalReference::libc_memchr_function(isolate()));
      Node* const result_address =
          CallCFunction3(MachineType::Pointer(), MachineType::Pointer(),
                         MachineType::IntPtr(), MachineType::UintPtr(), memchr,
                         string_addr, search_byte, search_length);
      GotoIf(WordEqual(result_address, int_zero), &return_minus_1);
      Node* const result_index =
          IntPtrAdd(IntPtrSub(result_address, string_addr), start_position);
      f_return(SmiTag(result_index));
    }

    BIND(&generic_fast_path);
    {
      Node* const result = CallSearchStringRaw<onebyte_t, onebyte_t>(
          adjusted_subject_ptr, subject_length, adjusted_search_ptr,
          search_length, start_position);
      f_return(SmiTag(result));
    }
  }

  BIND(&one_two);
  {
    Node* const adjusted_subject_ptr = PointerToStringDataAtIndex(
        subject_ptr, subject_offset, String::ONE_BYTE_ENCODING);
    Node* const adjusted_search_ptr = PointerToStringDataAtIndex(
        search_ptr, search_offset, String::TWO_BYTE_ENCODING);

    Node* const result = CallSearchStringRaw<onebyte_t, twobyte_t>(
        adjusted_subject_ptr, subject_length, adjusted_search_ptr,
        search_length, start_position);
    f_return(SmiTag(result));
  }

  BIND(&two_one);
  {
    Node* const adjusted_subject_ptr = PointerToStringDataAtIndex(
        subject_ptr, subject_offset, String::TWO_BYTE_ENCODING);
    Node* const adjusted_search_ptr = PointerToStringDataAtIndex(
        search_ptr, search_offset, String::ONE_BYTE_ENCODING);

    Node* const result = CallSearchStringRaw<twobyte_t, onebyte_t>(
        adjusted_subject_ptr, subject_length, adjusted_search_ptr,
        search_length, start_position);
    f_return(SmiTag(result));
  }

  BIND(&two_two);
  {
    Node* const adjusted_subject_ptr = PointerToStringDataAtIndex(
        subject_ptr, subject_offset, String::TWO_BYTE_ENCODING);
    Node* const adjusted_search_ptr = PointerToStringDataAtIndex(
        search_ptr, search_offset, String::TWO_BYTE_ENCODING);

    Node* const result = CallSearchStringRaw<twobyte_t, twobyte_t>(
        adjusted_subject_ptr, subject_length, adjusted_search_ptr,
        search_length, start_position);
    f_return(SmiTag(result));
  }

  BIND(&return_minus_1);
  f_return(SmiConstant(-1));

  BIND(&zero_length_needle);
  {
    Comment("0-length search_string");
    f_return(SmiTag(IntPtrMin(subject_length, start_position)));
  }

  BIND(&call_runtime_unchecked);
  {
    // Simplified version of the runtime call where the types of the arguments
    // are already known due to type checks in this stub.
    Comment("Call Runtime Unchecked");
    Node* result =
        CallRuntime(Runtime::kStringIndexOfUnchecked, NoContextConstant(),
                    subject_string, search_string, position);
    f_return(result);
  }
}
