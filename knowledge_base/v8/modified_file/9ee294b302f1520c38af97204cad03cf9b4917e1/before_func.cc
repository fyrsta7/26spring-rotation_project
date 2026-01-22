static Object* Runtime_NumberMod(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  CONVERT_DOUBLE_CHECKED(y, args[1]);

#ifdef WIN32
  // Workaround MS fmod bugs. ECMA-262 says:
  // dividend is finite and divisor is an infinity => result equals dividend
  // dividend is a zero and divisor is nonzero finite => result equals dividend
  if (!(isfinite(x) && (!isfinite(y) && !isnan(y))) &&
      !(x == 0 && (y != 0 && isfinite(y))))
#endif
  x = fmod(x, y);
  // NewNumberFromDouble may return a Smi instead of a Number object
  return Heap::NewNumberFromDouble(x);
}


static Object* Runtime_StringAdd(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_CHECKED(String, str1, args[0]);
  CONVERT_CHECKED(String, str2, args[1]);
  int len1 = str1->length();
  int len2 = str2->length();
  if (len1 == 0) return str2;
  if (len2 == 0) return str1;
  int length_sum = len1 + len2;
  // Make sure that an out of memory exception is thrown if the length
  // of the new cons string is too large to fit in a Smi.
  if (length_sum > Smi::kMaxValue || length_sum < 0) {
    Top::context()->mark_out_of_memory();
    return Failure::OutOfMemoryException();
  }
  return Heap::AllocateConsString(str1, str2);
}


template<typename sinkchar>
static inline void StringBuilderConcatHelper(String* special,
                                             StringShape special_shape,
                                             sinkchar* sink,
                                             FixedArray* fixed_array,
                                             int array_length) {
  int position = 0;
  for (int i = 0; i < array_length; i++) {
    Object* element = fixed_array->get(i);
    if (element->IsSmi()) {
      int len = Smi::cast(element)->value();
      int pos = len >> 11;
      len &= 0x7ff;
      String::WriteToFlat(special,
                          special_shape,
                          sink + position,
                          pos,
                          pos + len);
      position += len;
    } else {
      String* string = String::cast(element);
      StringShape shape(string);
      int element_length = string->length(shape);
      String::WriteToFlat(string, shape, sink + position, 0, element_length);
      position += element_length;
    }
  }
}


static Object* Runtime_StringBuilderConcat(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);
  CONVERT_CHECKED(JSArray, array, args[0]);
  CONVERT_CHECKED(String, special, args[1]);
  StringShape special_shape(special);
  int special_length = special->length(special_shape);
  Object* smi_array_length = array->length();
  if (!smi_array_length->IsSmi()) {
    Top::context()->mark_out_of_memory();
    return Failure::OutOfMemoryException();
  }
  int array_length = Smi::cast(smi_array_length)->value();
  if (!array->HasFastElements()) {
    return Top::Throw(Heap::illegal_argument_symbol());
  }
  FixedArray* fixed_array = FixedArray::cast(array->elements());
  if (fixed_array->length() < array_length) {
    array_length = fixed_array->length();
  }

  if (array_length == 0) {
    return Heap::empty_string();
  } else if (array_length == 1) {
    Object* first = fixed_array->get(0);
    if (first->IsString()) return first;
  }

  bool ascii = special_shape.IsAsciiRepresentation();
  int position = 0;
  for (int i = 0; i < array_length; i++) {
    Object* elt = fixed_array->get(i);
    if (elt->IsSmi()) {
      int len = Smi::cast(elt)->value();
      int pos = len >> 11;
      len &= 0x7ff;
      if (pos + len > special_length) {
        return Top::Throw(Heap::illegal_argument_symbol());
      }
      position += len;
    } else if (elt->IsString()) {
      String* element = String::cast(elt);
      StringShape element_shape(element);
      int element_length = element->length(element_shape);
      if (!Smi::IsValid(element_length + position)) {
        Top::context()->mark_out_of_memory();
        return Failure::OutOfMemoryException();
      }
      position += element_length;
      if (ascii && !element_shape.IsAsciiRepresentation()) {
        ascii = false;
      }
    } else {
      return Top::Throw(Heap::illegal_argument_symbol());
    }
  }

  int length = position;
  Object* object;

  if (ascii) {
    object = Heap::AllocateRawAsciiString(length);
    if (object->IsFailure()) return object;
    SeqAsciiString* answer = SeqAsciiString::cast(object);
    StringBuilderConcatHelper(special,
                              special_shape,
                              answer->GetChars(),
                              fixed_array,
                              array_length);
    return answer;
  } else {
    object = Heap::AllocateRawTwoByteString(length);
    if (object->IsFailure()) return object;
    SeqTwoByteString* answer = SeqTwoByteString::cast(object);
    StringBuilderConcatHelper(special,
                              special_shape,
                              answer->GetChars(),
                              fixed_array,
                              array_length);
    return answer;
  }
}


static Object* Runtime_NumberOr(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_NUMBER_CHECKED(int32_t, x, Int32, args[0]);
  CONVERT_NUMBER_CHECKED(int32_t, y, Int32, args[1]);
  return Heap::NumberFromInt32(x | y);
}


static Object* Runtime_NumberAnd(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_NUMBER_CHECKED(int32_t, x, Int32, args[0]);
  CONVERT_NUMBER_CHECKED(int32_t, y, Int32, args[1]);
  return Heap::NumberFromInt32(x & y);
}


static Object* Runtime_NumberXor(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_NUMBER_CHECKED(int32_t, x, Int32, args[0]);
  CONVERT_NUMBER_CHECKED(int32_t, y, Int32, args[1]);
  return Heap::NumberFromInt32(x ^ y);
}


static Object* Runtime_NumberNot(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_NUMBER_CHECKED(int32_t, x, Int32, args[0]);
  return Heap::NumberFromInt32(~x);
}


static Object* Runtime_NumberShl(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_NUMBER_CHECKED(int32_t, x, Int32, args[0]);
  CONVERT_NUMBER_CHECKED(int32_t, y, Int32, args[1]);
  return Heap::NumberFromInt32(x << (y & 0x1f));
}


static Object* Runtime_NumberShr(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_NUMBER_CHECKED(uint32_t, x, Uint32, args[0]);
  CONVERT_NUMBER_CHECKED(int32_t, y, Int32, args[1]);
  return Heap::NumberFromUint32(x >> (y & 0x1f));
}


static Object* Runtime_NumberSar(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_NUMBER_CHECKED(int32_t, x, Int32, args[0]);
  CONVERT_NUMBER_CHECKED(int32_t, y, Int32, args[1]);
  return Heap::NumberFromInt32(ArithmeticShiftRight(x, y & 0x1f));
}


static Object* Runtime_NumberEquals(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  CONVERT_DOUBLE_CHECKED(y, args[1]);
  if (isnan(x)) return Smi::FromInt(NOT_EQUAL);
  if (isnan(y)) return Smi::FromInt(NOT_EQUAL);
  if (x == y) return Smi::FromInt(EQUAL);
  Object* result;
  if ((fpclassify(x) == FP_ZERO) && (fpclassify(y) == FP_ZERO)) {
    result = Smi::FromInt(EQUAL);
  } else {
    result = Smi::FromInt(NOT_EQUAL);
  }
  return result;
}


static Object* Runtime_StringEquals(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_CHECKED(String, x, args[0]);
  CONVERT_CHECKED(String, y, args[1]);

  bool not_equal = !x->Equals(y);
  // This is slightly convoluted because the value that signifies
  // equality is 0 and inequality is 1 so we have to negate the result
  // from String::Equals.
  ASSERT(not_equal == 0 || not_equal == 1);
  STATIC_CHECK(EQUAL == 0);
  STATIC_CHECK(NOT_EQUAL == 1);
  return Smi::FromInt(not_equal);
}


static Object* Runtime_NumberCompare(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 3);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  CONVERT_DOUBLE_CHECKED(y, args[1]);
  if (isnan(x) || isnan(y)) return args[2];
  if (x == y) return Smi::FromInt(EQUAL);
  if (isless(x, y)) return Smi::FromInt(LESS);
  return Smi::FromInt(GREATER);
}


// Compare two Smis as if they were converted to strings and then
// compared lexicographically.
static Object* Runtime_SmiLexicographicCompare(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  // Arrays for the individual characters of the two Smis.  Smis are
  // 31 bit integers and 10 decimal digits are therefore enough.
  static int x_elms[10];
  static int y_elms[10];

  // Extract the integer values from the Smis.
  CONVERT_CHECKED(Smi, x, args[0]);
  CONVERT_CHECKED(Smi, y, args[1]);
  int x_value = x->value();
  int y_value = y->value();

  // If the integers are equal so are the string representations.
  if (x_value == y_value) return Smi::FromInt(EQUAL);

  // If one of the integers are zero the normal integer order is the
  // same as the lexicographic order of the string representations.
  if (x_value == 0 || y_value == 0) return Smi::FromInt(x_value - y_value);

  // If only one of the intergers is negative the negative number is
  // smallest because the char code of '-' is less than the char code
  // of any digit.  Otherwise, we make both values positive.
  if (x_value < 0 || y_value < 0) {
    if (y_value >= 0) return Smi::FromInt(LESS);
    if (x_value >= 0) return Smi::FromInt(GREATER);
    x_value = -x_value;
    y_value = -y_value;
  }

  // Convert the integers to arrays of their decimal digits.
  int x_index = 0;
  int y_index = 0;
  while (x_value > 0) {
    x_elms[x_index++] = x_value % 10;
    x_value /= 10;
  }
  while (y_value > 0) {
    y_elms[y_index++] = y_value % 10;
    y_value /= 10;
  }

  // Loop through the arrays of decimal digits finding the first place
  // where they differ.
  while (--x_index >= 0 && --y_index >= 0) {
    int diff = x_elms[x_index] - y_elms[y_index];
    if (diff != 0) return Smi::FromInt(diff);
  }

  // If one array is a suffix of the other array, the longest array is
  // the representation of the largest of the Smis in the
  // lexicographic ordering.
  return Smi::FromInt(x_index - y_index);
}


static Object* Runtime_StringCompare(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_CHECKED(String, x, args[0]);
  CONVERT_CHECKED(String, y, args[1]);

  StringShape x_shape(x);
  StringShape y_shape(y);

  // A few fast case tests before we flatten.
  if (x == y) return Smi::FromInt(EQUAL);
  if (y->length(y_shape) == 0) {
    if (x->length(x_shape) == 0) return Smi::FromInt(EQUAL);
    return Smi::FromInt(GREATER);
  } else if (x->length(x_shape) == 0) {
    return Smi::FromInt(LESS);
  }

  int d = x->Get(x_shape, 0) - y->Get(y_shape, 0);
  if (d < 0) return Smi::FromInt(LESS);
  else if (d > 0) return Smi::FromInt(GREATER);

  x->TryFlatten(x_shape);  // Shapes are no longer valid!
  y->TryFlatten(y_shape);

  static StringInputBuffer bufx;
  static StringInputBuffer bufy;
  bufx.Reset(x);
  bufy.Reset(y);
  while (bufx.has_more() && bufy.has_more()) {
    int d = bufx.GetNext() - bufy.GetNext();
    if (d < 0) return Smi::FromInt(LESS);
    else if (d > 0) return Smi::FromInt(GREATER);
  }

  // x is (non-trivial) prefix of y:
  if (bufy.has_more()) return Smi::FromInt(LESS);
  // y is prefix of x:
  return Smi::FromInt(bufx.has_more() ? GREATER : EQUAL);
}


static Object* Runtime_Math_abs(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(fabs(x));
}


static Object* Runtime_Math_acos(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(acos(x));
}


static Object* Runtime_Math_asin(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(asin(x));
}


static Object* Runtime_Math_atan(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(atan(x));
}


static Object* Runtime_Math_atan2(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  CONVERT_DOUBLE_CHECKED(y, args[1]);
  double result;
  if (isinf(x) && isinf(y)) {
    // Make sure that the result in case of two infinite arguments
    // is a multiple of Pi / 4. The sign of the result is determined
    // by the first argument (x) and the sign of the second argument
    // determines the multiplier: one or three.
    static double kPiDividedBy4 = 0.78539816339744830962;
    int multiplier = (x < 0) ? -1 : 1;
    if (y < 0) multiplier *= 3;
    result = multiplier * kPiDividedBy4;
  } else {
    result = atan2(x, y);
  }
  return Heap::AllocateHeapNumber(result);
}


static Object* Runtime_Math_ceil(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::NumberFromDouble(ceiling(x));
}


static Object* Runtime_Math_cos(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(cos(x));
}


static Object* Runtime_Math_exp(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(exp(x));
}


static Object* Runtime_Math_floor(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::NumberFromDouble(floor(x));
}


static Object* Runtime_Math_log(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(log(x));
}


static Object* Runtime_Math_pow(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 2);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  CONVERT_DOUBLE_CHECKED(y, args[1]);
  if (isnan(y) || ((x == 1 || x == -1) && isinf(y))) {
    return Heap::nan_value();
  } else if (y == 0) {
    return Smi::FromInt(1);
  } else {
    return Heap::AllocateHeapNumber(pow(x, y));
  }
}

// Returns a number value with positive sign, greater than or equal to
// 0 but less than 1, chosen randomly.
static Object* Runtime_Math_random(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 0);

  // To get much better precision, we combine the results of two
  // invocations of random(). The result is computed by normalizing a
  // double in the range [0, RAND_MAX + 1) obtained by adding the
  // high-order bits in the range [0, RAND_MAX] with the low-order
  // bits in the range [0, 1).
  double lo = static_cast<double>(random()) / (RAND_MAX + 1.0);
  double hi = static_cast<double>(random());
  double result = (hi + lo) / (RAND_MAX + 1.0);
  ASSERT(result >= 0 && result < 1);
  return Heap::AllocateHeapNumber(result);
}


static Object* Runtime_Math_round(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  if (signbit(x) && x >= -0.5) return Heap::minus_zero_value();
  return Heap::NumberFromDouble(floor(x + 0.5));
}


static Object* Runtime_Math_sin(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(sin(x));
}


static Object* Runtime_Math_sqrt(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(sqrt(x));
}


static Object* Runtime_Math_tan(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::AllocateHeapNumber(tan(x));
}


// The NewArguments function is only used when constructing the
// arguments array when calling non-functions from JavaScript in
// runtime.js:CALL_NON_FUNCTION.
static Object* Runtime_NewArguments(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  // ECMA-262, 3rd., 10.1.8, p.39
  CONVERT_CHECKED(JSFunction, callee, args[0]);

  // Compute the frame holding the arguments.
  JavaScriptFrameIterator it;
  it.AdvanceToArgumentsFrame();
  JavaScriptFrame* frame = it.frame();

  const int length = frame->GetProvidedParametersCount();
  Object* result = Heap::AllocateArgumentsObject(callee, length);
  if (result->IsFailure()) return result;
  if (length > 0) {
    Object* obj =  Heap::AllocateFixedArray(length);
    if (obj->IsFailure()) return obj;
    FixedArray* array = FixedArray::cast(obj);
    ASSERT(array->length() == length);
    WriteBarrierMode mode = array->GetWriteBarrierMode();
    for (int i = 0; i < length; i++) {
      array->set(i, frame->GetParameter(i), mode);
    }
    JSObject::cast(result)->set_elements(array);
  }
  return result;
}


static Object* Runtime_NewArgumentsFast(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 3);

  JSFunction* callee = JSFunction::cast(args[0]);
  Object** parameters = reinterpret_cast<Object**>(args[1]);
  const int length = Smi::cast(args[2])->value();

  Object* result = Heap::AllocateArgumentsObject(callee, length);
  if (result->IsFailure()) return result;
  ASSERT(Heap::InNewSpace(result));

  // Allocate the elements if needed.
  if (length > 0) {
    // Allocate the fixed array.
    Object* obj = Heap::AllocateRawFixedArray(length);
    if (obj->IsFailure()) return obj;
    reinterpret_cast<Array*>(obj)->set_map(Heap::fixed_array_map());
    FixedArray* array = FixedArray::cast(obj);
    array->set_length(length);
    WriteBarrierMode mode = array->GetWriteBarrierMode();
    for (int i = 0; i < length; i++) {
      array->set(i, *--parameters, mode);
    }
    JSObject::cast(result)->set_elements(FixedArray::cast(obj),
                                         SKIP_WRITE_BARRIER);
  }
  return result;
}


static Object* Runtime_NewClosure(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 2);
  CONVERT_ARG_CHECKED(JSFunction, boilerplate, 0);
  CONVERT_ARG_CHECKED(Context, context, 1);

  Handle<JSFunction> result =
      Factory::NewFunctionFromBoilerplate(boilerplate, context);
  return *result;
}


static Object* Runtime_NewObject(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  Object* constructor = args[0];
  if (constructor->IsJSFunction()) {
    JSFunction* function = JSFunction::cast(constructor);

    // Handle steping into constructors.
    if (Debug::StepInActive()) {
      StackFrameIterator it;
      it.Advance();
      ASSERT(it.frame()->is_construct());
      it.Advance();
      if (it.frame()->fp() == Debug::step_in_fp()) {
        HandleScope scope;
        Debug::FloodWithOneShot(Handle<SharedFunctionInfo>(function->shared()));
      }
    }

    if (function->has_initial_map() &&
        function->initial_map()->instance_type() == JS_FUNCTION_TYPE) {
      // The 'Function' function ignores the receiver object when
      // called using 'new' and creates a new JSFunction object that
      // is returned.  The receiver object is only used for error
      // reporting if an error occurs when constructing the new
      // JSFunction.  AllocateJSObject should not be used to allocate
      // JSFunctions since it does not properly initialize the shared
      // part of the function.  Since the receiver is ignored anyway,
      // we use the global object as the receiver instead of a new
      // JSFunction object.  This way, errors are reported the same
      // way whether or not 'Function' is called using 'new'.
      return Top::context()->global();
    }
    return Heap::AllocateJSObject(function);
  }

  HandleScope scope;
  Handle<Object> cons(constructor);
  // The constructor is not a function; throw a type error.
  Handle<Object> type_error =
    Factory::NewTypeError("not_constructor", HandleVector(&cons, 1));
  return Top::Throw(*type_error);
}


static Object* Runtime_LazyCompile(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);

  Handle<JSFunction> function = args.at<JSFunction>(0);
#ifdef DEBUG
  if (FLAG_trace_lazy) {
    PrintF("[lazy: ");
    function->shared()->name()->Print();
    PrintF("]\n");
  }
#endif

  // Compile the target function.
  ASSERT(!function->is_compiled());
  if (!CompileLazy(function, KEEP_EXCEPTION)) {
    return Failure::Exception();
  }

  return function->code();
}


static Object* Runtime_GetCalledFunction(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 0);
  StackFrameIterator it;
  // Get past the JS-to-C exit frame.
  ASSERT(it.frame()->is_exit());
  it.Advance();
  // Get past the CALL_NON_FUNCTION activation frame.
  ASSERT(it.frame()->is_java_script());
  it.Advance();
  // Argument adaptor frames do not copy the function; we have to skip
  // past them to get to the real calling frame.
  if (it.frame()->is_arguments_adaptor()) it.Advance();
  // Get the function from the top of the expression stack of the
  // calling frame.
  StandardFrame* frame = StandardFrame::cast(it.frame());
  int index = frame->ComputeExpressionsCount() - 1;
  Object* result = frame->GetExpression(index);
  return result;
}


static Object* Runtime_GetFunctionDelegate(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);
  RUNTIME_ASSERT(!args[0]->IsJSFunction());
  return *Execution::GetFunctionDelegate(args.at<Object>(0));
}


static Object* Runtime_NewContext(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_CHECKED(JSFunction, function, args[0]);
  int length = ScopeInfo<>::NumberOfContextSlots(function->code());
  Object* result = Heap::AllocateFunctionContext(length, function);
  if (result->IsFailure()) return result;

  Top::set_context(Context::cast(result));

  return result;  // non-failure
}


static Object* Runtime_PushContext(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  // Convert the object to a proper JavaScript object.
  Object* object = args[0];
  if (!object->IsJSObject()) {
    object = object->ToObject();
    if (object->IsFailure()) {
      if (!Failure::cast(object)->IsInternalError()) return object;
      HandleScope scope;
      Handle<Object> handle(args[0]);
      Handle<Object> result =
          Factory::NewTypeError("with_expression", HandleVector(&handle, 1));
      return Top::Throw(*result);
    }
  }

  Object* result =
      Heap::AllocateWithContext(Top::context(), JSObject::cast(object));
  if (result->IsFailure()) return result;

  Top::set_context(Context::cast(result));

  return result;
}


static Object* Runtime_LookupContext(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 2);

  CONVERT_ARG_CHECKED(Context, context, 0);
  CONVERT_ARG_CHECKED(String, name, 1);

  int index;
  PropertyAttributes attributes;
  ContextLookupFlags flags = FOLLOW_CHAINS;
  Handle<Object> holder =
      context->Lookup(name, flags, &index, &attributes);

  if (index < 0 && !holder.is_null()) {
    ASSERT(holder->IsJSObject());
    return *holder;
  }

  // No intermediate context found. Use global object by default.
  return Top::context()->global();
}


// A mechanism to return pairs of Object*'s. This is somewhat
// compiler-dependent as it assumes that a 64-bit value (a long long)
// is returned via two registers (edx:eax on ia32). Both the ia32 and
// arm platform support this; it is mostly an issue of "coaxing" the
// compiler to do the right thing.
//
// TODO(1236026): This is a non-portable hack that should be removed.
typedef uint64_t ObjectPair;
static inline ObjectPair MakePair(Object* x, Object* y) {
  return reinterpret_cast<uint32_t>(x) |
      (reinterpret_cast<ObjectPair>(y) << 32);
}


static inline Object* Unhole(Object* x, PropertyAttributes attributes) {
  ASSERT(!x->IsTheHole() || (attributes & READ_ONLY) != 0);
  USE(attributes);
  return x->IsTheHole() ? Heap::undefined_value() : x;
}


static JSObject* ComputeReceiverForNonGlobal(JSObject* holder) {
  ASSERT(!holder->IsGlobalObject());
  Context* top = Top::context();
  // Get the context extension function.
  JSFunction* context_extension_function =
      top->global_context()->context_extension_function();
  // If the holder isn't a context extension object, we just return it
  // as the receiver. This allows arguments objects to be used as
  // receivers, but only if they are put in the context scope chain
  // explicitly via a with-statement.
  Object* constructor = holder->map()->constructor();
  if (constructor != context_extension_function) return holder;
  // Fall back to using the global object as the receiver if the
  // property turns out to be a local variable allocated in a context
  // extension object - introduced via eval.
  return top->global()->global_receiver();
}


static ObjectPair LoadContextSlotHelper(Arguments args, bool throw_error) {
  HandleScope scope;
  ASSERT(args.length() == 2);

  if (!args[0]->IsContext() || !args[1]->IsString()) {
    return MakePair(IllegalOperation(), NULL);
  }
  Handle<Context> context = args.at<Context>(0);
  Handle<String> name = args.at<String>(1);

  int index;
  PropertyAttributes attributes;
  ContextLookupFlags flags = FOLLOW_CHAINS;
  Handle<Object> holder =
      context->Lookup(name, flags, &index, &attributes);

  // If the index is non-negative, the slot has been found in a local
  // variable or a parameter. Read it from the context object or the
  // arguments object.
  if (index >= 0) {
    // If the "property" we were looking for is a local variable or an
    // argument in a context, the receiver is the global object; see
    // ECMA-262, 3rd., 10.1.6 and 10.2.3.
    JSObject* receiver = Top::context()->global()->global_receiver();
    Object* value = (holder->IsContext())
        ? Context::cast(*holder)->get(index)
        : JSObject::cast(*holder)->GetElement(index);
    return MakePair(Unhole(value, attributes), receiver);
  }

  // If the holder is found, we read the property from it.
  if (!holder.is_null() && holder->IsJSObject()) {
    ASSERT(Handle<JSObject>::cast(holder)->HasProperty(*name));
    JSObject* object = JSObject::cast(*holder);
    JSObject* receiver = (object->IsGlobalObject())
        ? GlobalObject::cast(object)->global_receiver()
        : ComputeReceiverForNonGlobal(object);
    // No need to unhole the value here. This is taken care of by the
    // GetProperty function.
    Object* value = object->GetProperty(*name);
    return MakePair(value, receiver);
  }

  if (throw_error) {
    // The property doesn't exist - throw exception.
    Handle<Object> reference_error =
        Factory::NewReferenceError("not_defined", HandleVector(&name, 1));
    return MakePair(Top::Throw(*reference_error), NULL);
  } else {
    // The property doesn't exist - return undefined
    return MakePair(Heap::undefined_value(), Heap::undefined_value());
  }
}


static ObjectPair Runtime_LoadContextSlot(Arguments args) {
  return LoadContextSlotHelper(args, true);
}


static ObjectPair Runtime_LoadContextSlotNoReferenceError(Arguments args) {
  return LoadContextSlotHelper(args, false);
}


static Object* Runtime_StoreContextSlot(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 3);

  Handle<Object> value(args[0]);
  CONVERT_ARG_CHECKED(Context, context, 1);
  CONVERT_ARG_CHECKED(String, name, 2);

  int index;
  PropertyAttributes attributes;
  ContextLookupFlags flags = FOLLOW_CHAINS;
  Handle<Object> holder =
      context->Lookup(name, flags, &index, &attributes);

  if (index >= 0) {
    if (holder->IsContext()) {
      // Ignore if read_only variable.
      if ((attributes & READ_ONLY) == 0) {
        Handle<Context>::cast(holder)->set(index, *value);
      }
    } else {
      ASSERT((attributes & READ_ONLY) == 0);
      Object* result =
          Handle<JSObject>::cast(holder)->SetElement(index, *value);
      USE(result);
      ASSERT(!result->IsFailure());
    }
    return *value;
  }

  // Slow case: The property is not in a FixedArray context.
  // It is either in an JSObject extension context or it was not found.
  Handle<JSObject> context_ext;

  if (!holder.is_null()) {
    // The property exists in the extension context.
    context_ext = Handle<JSObject>::cast(holder);
  } else {
    // The property was not found. It needs to be stored in the global context.
    ASSERT(attributes == ABSENT);
    attributes = NONE;
    context_ext = Handle<JSObject>(Top::context()->global());
  }

  // Set the property, but ignore if read_only variable.
  if ((attributes & READ_ONLY) == 0) {
    Handle<Object> set = SetProperty(context_ext, name, value, attributes);
    if (set.is_null()) {
      // Failure::Exception is converted to a null handle in the
      // handle-based methods such as SetProperty.  We therefore need
      // to convert null handles back to exceptions.
      ASSERT(Top::has_pending_exception());
      return Failure::Exception();
    }
  }
  return *value;
}


static Object* Runtime_Throw(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);

  return Top::Throw(args[0]);
}


static Object* Runtime_ReThrow(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);

  return Top::ReThrow(args[0]);
}


static Object* Runtime_ThrowReferenceError(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);

  Handle<Object> name(args[0]);
  Handle<Object> reference_error =
    Factory::NewReferenceError("not_defined", HandleVector(&name, 1));
  return Top::Throw(*reference_error);
}


static Object* Runtime_StackOverflow(Arguments args) {
  NoHandleAllocation na;
  return Top::StackOverflow();
}


static Object* RuntimePreempt(Arguments args) {
  // Clear the preempt request flag.
  StackGuard::Continue(PREEMPT);

  ContextSwitcher::PreemptionReceived();

  {
    v8::Unlocker unlocker;
    Thread::YieldCPU();
  }

  return Heap::undefined_value();
}


static Object* DebugBreakHelper() {
  // Just continue if breaks are disabled.
  if (Debug::disable_break()) {
    return Heap::undefined_value();
  }

  // Don't break in system functions. If the current function is
  // either in the builtins object of some context or is in the debug
  // context just return with the debug break stack guard active.
  JavaScriptFrameIterator it;
  JavaScriptFrame* frame = it.frame();
  Object* fun = frame->function();
  if (fun->IsJSFunction()) {
    GlobalObject* global = JSFunction::cast(fun)->context()->global();
    if (global->IsJSBuiltinsObject() || Debug::IsDebugGlobal(global)) {
      return Heap::undefined_value();
    }
  }

  // Clear the debug request flag.
  StackGuard::Continue(DEBUGBREAK);

  HandleScope scope;
  // Enter the debugger. Just continue if we fail to enter the debugger.
  EnterDebugger debugger;
  if (debugger.FailedToEnter()) {
    return Heap::undefined_value();
  }

  // Notify the debug event listeners.
  Debugger::OnDebugBreak(Factory::undefined_value());

  // Return to continue execution.
  return Heap::undefined_value();
}


static Object* Runtime_DebugBreak(Arguments args) {
  ASSERT(args.length() == 0);
  return DebugBreakHelper();
}


static Object* Runtime_StackGuard(Arguments args) {
  ASSERT(args.length() == 1);

  // First check if this is a real stack overflow.
  if (StackGuard::IsStackOverflow()) return Runtime_StackOverflow(args);

  // If not real stack overflow the stack guard was used to interrupt
  // execution for another purpose.
  if (StackGuard::IsDebugBreak()) DebugBreakHelper();
  if (StackGuard::IsPreempted()) RuntimePreempt(args);
  if (StackGuard::IsInterrupted()) {
    // interrupt
    StackGuard::Continue(INTERRUPT);
    return Top::StackOverflow();
  }
  return Heap::undefined_value();
}


// NOTE: These PrintXXX functions are defined for all builds (not just
// DEBUG builds) because we may want to be able to trace function
// calls in all modes.
static void PrintString(String* str) {
  // not uncommon to have empty strings
  if (str->length() > 0) {
    SmartPointer<char> s =
        str->ToCString(DISALLOW_NULLS, ROBUST_STRING_TRAVERSAL);
    PrintF("%s", *s);
  }
}


static void PrintObject(Object* obj) {
  if (obj->IsSmi()) {
    PrintF("%d", Smi::cast(obj)->value());
  } else if (obj->IsString() || obj->IsSymbol()) {
    PrintString(String::cast(obj));
  } else if (obj->IsNumber()) {
    PrintF("%g", obj->Number());
  } else if (obj->IsFailure()) {
    PrintF("<failure>");
  } else if (obj->IsUndefined()) {
    PrintF("<undefined>");
  } else if (obj->IsNull()) {
    PrintF("<null>");
  } else if (obj->IsTrue()) {
    PrintF("<true>");
  } else if (obj->IsFalse()) {
    PrintF("<false>");
  } else {
    PrintF("%p", obj);
  }
}


static int StackSize() {
  int n = 0;
  for (JavaScriptFrameIterator it; !it.done(); it.Advance()) n++;
  return n;
}


static void PrintTransition(Object* result) {
  // indentation
  { const int nmax = 80;
    int n = StackSize();
    if (n <= nmax)
      PrintF("%4d:%*s", n, n, "");
    else
      PrintF("%4d:%*s", n, nmax, "...");
  }

  if (result == NULL) {
    // constructor calls
    JavaScriptFrameIterator it;
    JavaScriptFrame* frame = it.frame();
    if (frame->IsConstructor()) PrintF("new ");
    // function name
    Object* fun = frame->function();
    if (fun->IsJSFunction()) {
      PrintObject(JSFunction::cast(fun)->shared()->name());
    } else {
      PrintObject(fun);
    }
    // function arguments
    // (we are intentionally only printing the actually
    // supplied parameters, not all parameters required)
    PrintF("(this=");
    PrintObject(frame->receiver());
    const int length = frame->GetProvidedParametersCount();
    for (int i = 0; i < length; i++) {
      PrintF(", ");
      PrintObject(frame->GetParameter(i));
    }
    PrintF(") {\n");

  } else {
    // function result
    PrintF("} -> ");
    PrintObject(result);
    PrintF("\n");
  }
}


static Object* Runtime_TraceEnter(Arguments args) {
  ASSERT(args.length() == 0);
  NoHandleAllocation ha;
  PrintTransition(NULL);
  return Heap::undefined_value();
}


static Object* Runtime_TraceExit(Arguments args) {
  NoHandleAllocation ha;
  PrintTransition(args[0]);
  return args[0];  // return TOS
}


static Object* Runtime_DebugPrint(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

#ifdef DEBUG
  if (args[0]->IsString()) {
    // If we have a string, assume it's a code "marker"
    // and print some interesting cpu debugging info.
    JavaScriptFrameIterator it;
    JavaScriptFrame* frame = it.frame();
    PrintF("fp = %p, sp = %p, pp = %p: ",
           frame->fp(), frame->sp(), frame->pp());
  } else {
    PrintF("DebugPrint: ");
  }
  args[0]->Print();
#else
  // ShortPrint is available in release mode. Print is not.
  args[0]->ShortPrint();
#endif
  PrintF("\n");
  Flush();

  return args[0];  // return TOS
}


static Object* Runtime_DebugTrace(Arguments args) {
  ASSERT(args.length() == 0);
  NoHandleAllocation ha;
  Top::PrintStack();
  return Heap::undefined_value();
}


static Object* Runtime_DateCurrentTime(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 0);

  // According to ECMA-262, section 15.9.1, page 117, the precision of
  // the number in a Date object representing a particular instant in
  // time is milliseconds. Therefore, we floor the result of getting
  // the OS time.
  double millis = floor(OS::TimeCurrentMillis());
  return Heap::NumberFromDouble(millis);
}


static Object* Runtime_DateParseString(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);

  CONVERT_CHECKED(String, string_object, args[0]);

  Handle<String> str(string_object);
  Handle<FixedArray> output = Factory::NewFixedArray(DateParser::OUTPUT_SIZE);
  if (DateParser::Parse(*str, *output)) {
    return *Factory::NewJSArrayWithElements(output);
  } else {
    return *Factory::null_value();
  }
}


static Object* Runtime_DateLocalTimezone(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  char* zone = OS::LocalTimezone(x);
  return Heap::AllocateStringFromUtf8(CStrVector(zone));
}


static Object* Runtime_DateLocalTimeOffset(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 0);

  return Heap::NumberFromDouble(OS::LocalTimeOffset());
}


static Object* Runtime_DateDaylightSavingsOffset(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(x, args[0]);
  return Heap::NumberFromDouble(OS::DaylightSavingsOffset(x));
}


static Object* Runtime_NumberIsFinite(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_DOUBLE_CHECKED(value, args[0]);
  Object* result;
  if (isnan(value) || (fpclassify(value) == FP_INFINITE)) {
    result = Heap::false_value();
  } else {
    result = Heap::true_value();
  }
  return result;
}


static Object* EvalContext() {
  // The topmost JS frame belongs to the eval function which called
  // the CompileString runtime function. We need to unwind one level
  // to get to the caller of eval.
  StackFrameLocator locator;
  JavaScriptFrame* frame = locator.FindJavaScriptFrame(1);

  // TODO(900055): Right now we check if the caller of eval() supports
  // eval to determine if it's an aliased eval or not. This may not be
  // entirely correct in the unlikely case where a function uses both
  // aliased and direct eval calls.
  HandleScope scope;
  if (!ScopeInfo<>::SupportsEval(frame->FindCode())) {
    // Aliased eval: Evaluate in the global context of the eval
    // function to support aliased, cross environment evals.
    return *Top::global_context();
  }

  // Fetch the caller context from the frame.
  Handle<Context> caller(Context::cast(frame->context()));

  // Check for eval() invocations that cross environments. Use the
  // context from the stack if evaluating in current environment.
  Handle<Context> target = Top::global_context();
  if (caller->global_context() == *target) return *caller;

  // Otherwise, use the global context from the other environment.
  return *target;
}


static Object* Runtime_EvalReceiver(Arguments args) {
  ASSERT(args.length() == 1);
  StackFrameLocator locator;
  JavaScriptFrame* frame = locator.FindJavaScriptFrame(1);
  // Fetch the caller context from the frame.
  Context* caller = Context::cast(frame->context());

  // Check for eval() invocations that cross environments. Use the
  // top frames receiver if evaluating in current environment.
  Context* global_context = Top::context()->global()->global_context();
  if (caller->global_context() == global_context) {
    return frame->receiver();
  }

  // Otherwise use the given argument (the global object of the
  // receiving context).
  return args[0];
}


static Object* Runtime_GlobalReceiver(Arguments args) {
  ASSERT(args.length() == 1);
  Object* global = args[0];
  if (!global->IsJSGlobalObject()) return Heap::null_value();
  return JSGlobalObject::cast(global)->global_receiver();
}


static Object* Runtime_CompileString(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 3);
  CONVERT_ARG_CHECKED(String, source, 0);
  CONVERT_ARG_CHECKED(Smi, line_offset, 1);
  bool contextual = args[2]->IsTrue();
  RUNTIME_ASSERT(contextual || args[2]->IsFalse());

  // Compute the eval context.
  Handle<Context> context;
  if (contextual) {
    // Get eval context. May not be available if we are calling eval
    // through an alias, and the corresponding frame doesn't have a
    // proper eval context set up.
    Object* eval_context = EvalContext();
    if (eval_context->IsFailure()) return eval_context;
    context = Handle<Context>(Context::cast(eval_context));
  } else {
    context = Handle<Context>(Top::context()->global_context());
  }


  // Compile source string.
  bool is_global = context->IsGlobalContext();
  Handle<JSFunction> boilerplate =
      Compiler::CompileEval(source, line_offset->value(), is_global);
  if (boilerplate.is_null()) return Failure::Exception();
  Handle<JSFunction> fun =
      Factory::NewFunctionFromBoilerplate(boilerplate, context);
  return *fun;
}


static Object* Runtime_CompileScript(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 4);

  CONVERT_ARG_CHECKED(String, source, 0);
  CONVERT_ARG_CHECKED(String, script, 1);
  CONVERT_CHECKED(Smi, line_attrs, args[2]);
  int line = line_attrs->value();
  CONVERT_CHECKED(Smi, col_attrs, args[3]);
  int col = col_attrs->value();
  Handle<JSFunction> boilerplate =
      Compiler::Compile(source, script, line, col, NULL, NULL);
  if (boilerplate.is_null()) return Failure::Exception();
  Handle<JSFunction> fun =
      Factory::NewFunctionFromBoilerplate(boilerplate,
                                          Handle<Context>(Top::context()));
  return *fun;
}


static Object* Runtime_SetNewFunctionAttributes(Arguments args) {
  // This utility adjusts the property attributes for newly created Function
  // object ("new Function(...)") by changing the map.
  // All it does is changing the prototype property to enumerable
  // as specified in ECMA262, 15.3.5.2.
  HandleScope scope;
  ASSERT(args.length() == 1);
  CONVERT_ARG_CHECKED(JSFunction, func, 0);
  ASSERT(func->map()->instance_type() ==
         Top::function_instance_map()->instance_type());
  ASSERT(func->map()->instance_size() ==
         Top::function_instance_map()->instance_size());
  func->set_map(*Top::function_instance_map());
  return *func;
}


// Push an array unto an array of arrays if it is not already in the
// array.  Returns true if the element was pushed on the stack and
// false otherwise.
static Object* Runtime_PushIfAbsent(Arguments args) {
  ASSERT(args.length() == 2);
  CONVERT_CHECKED(JSArray, array, args[0]);
  CONVERT_CHECKED(JSArray, element, args[1]);
  RUNTIME_ASSERT(array->HasFastElements());
  int length = Smi::cast(array->length())->value();
  FixedArray* elements = FixedArray::cast(array->elements());
  for (int i = 0; i < length; i++) {
    if (elements->get(i) == element) return Heap::false_value();
  }
  Object* obj = array->SetFastElement(length, element);
  if (obj->IsFailure()) return obj;
  return Heap::true_value();
}


/**
 * A simple visitor visits every element of Array's.
 * The backend storage can be a fixed array for fast elements case,
 * or a dictionary for sparse array. Since Dictionary is a subtype
 * of FixedArray, the class can be used by both fast and slow cases.
 * The second parameter of the constructor, fast_elements, specifies
 * whether the storage is a FixedArray or Dictionary.
 *
 * An index limit is used to deal with the situation that a result array
 * length overflows 32-bit non-negative integer.
 */
class ArrayConcatVisitor {
 public:
  ArrayConcatVisitor(Handle<FixedArray> storage,
                     uint32_t index_limit,
                     bool fast_elements) :
      storage_(storage), index_limit_(index_limit),
      fast_elements_(fast_elements), index_offset_(0) { }

  void visit(uint32_t i, Handle<Object> elm) {
    uint32_t index = i + index_offset_;
    if (index >= index_limit_) return;

    if (fast_elements_) {
      ASSERT(index < static_cast<uint32_t>(storage_->length()));
      storage_->set(index, *elm);

    } else {
      Handle<Dictionary> dict = Handle<Dictionary>::cast(storage_);
      Handle<Dictionary> result =
          Factory::DictionaryAtNumberPut(dict, index, elm);
      if (!result.is_identical_to(dict))
        storage_ = result;
    }
  }

  void increase_index_offset(uint32_t delta) {
    index_offset_ += delta;
  }

 private:
  Handle<FixedArray> storage_;
  uint32_t index_limit_;
  bool fast_elements_;
  uint32_t index_offset_;
};


/**
 * A helper function that visits elements of a JSObject. Only elements
 * whose index between 0 and range (exclusive) are visited.
 *
 * If the third parameter, visitor, is not NULL, the visitor is called
 * with parameters, 'visitor_index_offset + element index' and the element.
 *
 * It returns the number of visisted elements.
 */
static uint32_t IterateElements(Handle<JSObject> receiver,
                                uint32_t range,
                                ArrayConcatVisitor* visitor) {
  uint32_t num_of_elements = 0;

  if (receiver->HasFastElements()) {
    Handle<FixedArray> elements(FixedArray::cast(receiver->elements()));
    uint32_t len = elements->length();
    if (range < len) len = range;

    for (uint32_t j = 0; j < len; j++) {
      Handle<Object> e(elements->get(j));
      if (!e->IsTheHole()) {
        num_of_elements++;
        if (visitor)
          visitor->visit(j, e);
      }
    }

  } else {
    Handle<Dictionary> dict(receiver->element_dictionary());
    uint32_t capacity = dict->Capacity();
    for (uint32_t j = 0; j < capacity; j++) {
      Handle<Object> k(dict->KeyAt(j));
      if (dict->IsKey(*k)) {
        ASSERT(k->IsNumber());
        uint32_t index = static_cast<uint32_t>(k->Number());
        if (index < range) {
          num_of_elements++;
          if (visitor) {
            visitor->visit(index,
                           Handle<Object>(dict->ValueAt(j)));
          }
        }
      }
    }
  }

  return num_of_elements;
}


/**
 * A helper function that visits elements of an Array object, and elements
 * on its prototypes.
 *
 * Elements on prototypes are visited first, and only elements whose indices
 * less than Array length are visited.
 *
 * If a ArrayConcatVisitor object is given, the visitor is called with
 * parameters, element's index + visitor_index_offset and the element.
 */
static uint32_t IterateArrayAndPrototypeElements(Handle<JSArray> array,
                                                 ArrayConcatVisitor* visitor) {
  uint32_t range = static_cast<uint32_t>(array->length()->Number());
  Handle<Object> obj = array;

  static const int kEstimatedPrototypes = 3;
  List< Handle<JSObject> > objects(kEstimatedPrototypes);

  // Visit prototype first. If an element on the prototype is shadowed by
  // the inheritor using the same index, the ArrayConcatVisitor visits
  // the prototype element before the shadowing element.
  // The visitor can simply overwrite the old value by new value using
  // the same index.  This follows Array::concat semantics.
  while (!obj->IsNull()) {
    objects.Add(Handle<JSObject>::cast(obj));
    obj = Handle<Object>(obj->GetPrototype());
  }

  uint32_t nof_elements = 0;
  for (int i = objects.length() - 1; i >= 0; i--) {
    Handle<JSObject> obj = objects[i];
    nof_elements +=
        IterateElements(Handle<JSObject>::cast(obj), range, visitor);
  }

  return nof_elements;
}


/**
 * A helper function of Runtime_ArrayConcat.
 *
 * The first argument is an Array of arrays and objects. It is the
 * same as the arguments array of Array::concat JS function.
 *
 * If an argument is an Array object, the function visits array
 * elements.  If an argument is not an Array object, the function
 * visits the object as if it is an one-element array.
 *
 * If the result array index overflows 32-bit integer, the rounded
 * non-negative number is used as new length. For example, if one
 * array length is 2^32 - 1, second array length is 1, the
 * concatenated array length is 0.
 */
static uint32_t IterateArguments(Handle<JSArray> arguments,
                                 ArrayConcatVisitor* visitor) {
  uint32_t visited_elements = 0;
  uint32_t num_of_args = static_cast<uint32_t>(arguments->length()->Number());

  for (uint32_t i = 0; i < num_of_args; i++) {
    Handle<Object> obj(arguments->GetElement(i));
    if (obj->IsJSArray()) {
      Handle<JSArray> array = Handle<JSArray>::cast(obj);
      uint32_t len = static_cast<uint32_t>(array->length()->Number());
      uint32_t nof_elements =
          IterateArrayAndPrototypeElements(array, visitor);
      // Total elements of array and its prototype chain can be more than
      // the array length, but ArrayConcat can only concatenate at most
      // the array length number of elements.
      visited_elements += (nof_elements > len) ? len : nof_elements;
      if (visitor) visitor->increase_index_offset(len);

    } else {
      if (visitor) {
        visitor->visit(0, obj);
        visitor->increase_index_offset(1);
      }
      visited_elements++;
    }
  }
  return visited_elements;
}


/**
 * Array::concat implementation.
 * See ECMAScript 262, 15.4.4.4.
 */
static Object* Runtime_ArrayConcat(Arguments args) {
  ASSERT(args.length() == 1);
  HandleScope handle_scope;

  CONVERT_CHECKED(JSArray, arg_arrays, args[0]);
  Handle<JSArray> arguments(arg_arrays);

  // Pass 1: estimate the number of elements of the result
  // (it could be more than real numbers if prototype has elements).
  uint32_t result_length = 0;
  uint32_t num_of_args = static_cast<uint32_t>(arguments->length()->Number());

  { AssertNoAllocation nogc;
    for (uint32_t i = 0; i < num_of_args; i++) {
      Object* obj = arguments->GetElement(i);
      if (obj->IsJSArray()) {
        result_length +=
            static_cast<uint32_t>(JSArray::cast(obj)->length()->Number());
      } else {
        result_length++;
      }
    }
  }

  // Allocate an empty array, will set length and content later.
  Handle<JSArray> result = Factory::NewJSArray(0);

  uint32_t estimate_nof_elements = IterateArguments(arguments, NULL);
  // If estimated number of elements is more than half of length, a
  // fixed array (fast case) is more time and space-efficient than a
  // dictionary.
  bool fast_case = (estimate_nof_elements * 2) >= result_length;

  Handle<FixedArray> storage;
  if (fast_case) {
    // The backing storage array must have non-existing elements to
    // preserve holes across concat operations.
    storage = Factory::NewFixedArrayWithHoles(result_length);

  } else {
    // TODO(126): move 25% pre-allocation logic into Dictionary::Allocate
    uint32_t at_least_space_for = estimate_nof_elements +
                                  (estimate_nof_elements >> 2);
    storage = Handle<FixedArray>::cast(
                  Factory::NewDictionary(at_least_space_for));
  }

  Handle<Object> len = Factory::NewNumber(static_cast<double>(result_length));

  ArrayConcatVisitor visitor(storage, result_length, fast_case);

  IterateArguments(arguments, &visitor);

  result->set_length(*len);
  result->set_elements(*storage);

  return *result;
}


// This will not allocate (flatten the string), but it may run
// very slowly for very deeply nested ConsStrings.  For debugging use only.
static Object* Runtime_GlobalPrint(Arguments args) {
  NoHandleAllocation ha;
  ASSERT(args.length() == 1);

  CONVERT_CHECKED(String, string, args[0]);
  StringInputBuffer buffer(string);
  while (buffer.has_more()) {
    uint16_t character = buffer.GetNext();
    PrintF("%c", character);
  }
  return string;
}


static Object* Runtime_RemoveArrayHoles(Arguments args) {
  ASSERT(args.length() == 1);
  // Ignore the case if this is not a JSArray.
  if (!args[0]->IsJSArray()) return args[0];
  return JSArray::cast(args[0])->RemoveHoles();
}


// Move contents of argument 0 (an array) to argument 1 (an array)
static Object* Runtime_MoveArrayContents(Arguments args) {
  ASSERT(args.length() == 2);
  CONVERT_CHECKED(JSArray, from, args[0]);
  CONVERT_CHECKED(JSArray, to, args[1]);
  to->SetContent(FixedArray::cast(from->elements()));
  to->set_length(from->length());
  from->SetContent(Heap::empty_fixed_array());
  from->set_length(0);
  return to;
}


// How many elements does this array have?
static Object* Runtime_EstimateNumberOfElements(Arguments args) {
  ASSERT(args.length() == 1);
  CONVERT_CHECKED(JSArray, array, args[0]);
  HeapObject* elements = array->elements();
  if (elements->IsDictionary()) {
    return Smi::FromInt(Dictionary::cast(elements)->NumberOfElements());
  } else {
    return array->length();
  }
}


// Returns an array that tells you where in the [0, length) interval an array
// might have elements.  Can either return keys or intervals.  Keys can have
// gaps in (undefined).  Intervals can also span over some undefined keys.
static Object* Runtime_GetArrayKeys(Arguments args) {
  ASSERT(args.length() == 2);
  HandleScope scope;
  CONVERT_CHECKED(JSArray, raw_array, args[0]);
  Handle<JSArray> array(raw_array);
  CONVERT_NUMBER_CHECKED(uint32_t, length, Uint32, args[1]);
  if (array->elements()->IsDictionary()) {
    // Create an array and get all the keys into it, then remove all the
    // keys that are not integers in the range 0 to length-1.
    Handle<FixedArray> keys = GetKeysInFixedArrayFor(array);
    int keys_length = keys->length();
    for (int i = 0; i < keys_length; i++) {
      Object* key = keys->get(i);
      uint32_t index;
      if (!Array::IndexFromObject(key, &index) || index >= length) {
        // Zap invalid keys.
        keys->set_undefined(i);
      }
    }
    return *Factory::NewJSArrayWithElements(keys);
  } else {
    Handle<FixedArray> single_interval = Factory::NewFixedArray(2);
    // -1 means start of array.
    single_interval->set(0,
                         Smi::FromInt(-1),
                         SKIP_WRITE_BARRIER);
    Handle<Object> length_object =
        Factory::NewNumber(static_cast<double>(length));
    single_interval->set(1, *length_object);
    return *Factory::NewJSArrayWithElements(single_interval);
  }
}


// DefineAccessor takes an optional final argument which is the
// property attributes (eg, DONT_ENUM, DONT_DELETE).  IMPORTANT: due
// to the way accessors are implemented, it is set for both the getter
// and setter on the first call to DefineAccessor and ignored on
// subsequent calls.
static Object* Runtime_DefineAccessor(Arguments args) {
  RUNTIME_ASSERT(args.length() == 4 || args.length() == 5);
  // Compute attributes.
  PropertyAttributes attributes = NONE;
  if (args.length() == 5) {
    CONVERT_CHECKED(Smi, attrs, args[4]);
    int value = attrs->value();
    // Only attribute bits should be set.
    ASSERT((value & ~(READ_ONLY | DONT_ENUM | DONT_DELETE)) == 0);
    attributes = static_cast<PropertyAttributes>(value);
  }

  CONVERT_CHECKED(JSObject, obj, args[0]);
  CONVERT_CHECKED(String, name, args[1]);
  CONVERT_CHECKED(Smi, flag, args[2]);
  CONVERT_CHECKED(JSFunction, fun, args[3]);
  return obj->DefineAccessor(name, flag->value() == 0, fun, attributes);
}


static Object* Runtime_LookupAccessor(Arguments args) {
  ASSERT(args.length() == 3);
  CONVERT_CHECKED(JSObject, obj, args[0]);
  CONVERT_CHECKED(String, name, args[1]);
  CONVERT_CHECKED(Smi, flag, args[2]);
  return obj->LookupAccessor(name, flag->value() == 0);
}


// Helper functions for wrapping and unwrapping stack frame ids.
static Smi* WrapFrameId(StackFrame::Id id) {
  ASSERT(IsAligned(OffsetFrom(id), 4));
  return Smi::FromInt(id >> 2);
}


static StackFrame::Id UnwrapFrameId(Smi* wrapped) {
  return static_cast<StackFrame::Id>(wrapped->value() << 2);
}


// Adds a JavaScript function as a debug event listener.
// args[0]: debug event listener function
// args[1]: object supplied during callback
static Object* Runtime_AddDebugEventListener(Arguments args) {
  ASSERT(args.length() == 2);
  // Convert the parameters to API objects to call the API function for adding
  // a JavaScript function as debug event listener.
  CONVERT_ARG_CHECKED(JSFunction, raw_fun, 0);
  v8::Handle<v8::Function> fun(ToApi<v8::Function>(raw_fun));
  v8::Handle<v8::Value> data(ToApi<v8::Value>(args.at<Object>(0)));
  v8::Debug::AddDebugEventListener(fun, data);

  return Heap::undefined_value();
}


// Removes a JavaScript function debug event listener.
// args[0]: debug event listener function
static Object* Runtime_RemoveDebugEventListener(Arguments args) {
  ASSERT(args.length() == 1);
  // Convert the parameter to an API object to call the API function for
  // removing a JavaScript function debug event listener.
  CONVERT_ARG_CHECKED(JSFunction, raw_fun, 0);
  v8::Handle<v8::Function> fun(ToApi<v8::Function>(raw_fun));
  v8::Debug::RemoveDebugEventListener(fun);

  return Heap::undefined_value();
}


static Object* Runtime_Break(Arguments args) {
  ASSERT(args.length() == 0);
  StackGuard::DebugBreak();
  return Heap::undefined_value();
}


static Object* DebugLookupResultValue(LookupResult* result) {
  Object* value;
  switch (result->type()) {
    case NORMAL: {
      Dictionary* dict =
          JSObject::cast(result->holder())->property_dictionary();
      value = dict->ValueAt(result->GetDictionaryEntry());
      if (value->IsTheHole()) {
        return Heap::undefined_value();
      }
      return value;
    }
    case FIELD:
      value =
          JSObject::cast(
              result->holder())->FastPropertyAt(result->GetFieldIndex());
      if (value->IsTheHole()) {
        return Heap::undefined_value();
      }
      return value;
    case CONSTANT_FUNCTION:
      return result->GetConstantFunction();
    case CALLBACKS:
    case INTERCEPTOR:
    case MAP_TRANSITION:
    case CONSTANT_TRANSITION:
    case NULL_DESCRIPTOR:
      return Heap::undefined_value();
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return Heap::undefined_value();
}


static Object* Runtime_DebugGetPropertyDetails(Arguments args) {
  HandleScope scope;

  ASSERT(args.length() == 2);

  CONVERT_ARG_CHECKED(JSObject, obj, 0);
  CONVERT_ARG_CHECKED(String, name, 1);

  // Check if the name is trivially convertible to an index and get the element
  // if so.
  uint32_t index;
  if (name->AsArrayIndex(&index)) {
    Handle<FixedArray> details = Factory::NewFixedArray(2);
    details->set(0, Runtime::GetElementOrCharAt(obj, index));
    details->set(1, PropertyDetails(NONE, NORMAL).AsSmi());
    return *Factory::NewJSArrayWithElements(details);
  }

  // Perform standard local lookup on the object.
  LookupResult result;
  obj->Lookup(*name, &result);
  if (result.IsProperty()) {
    Handle<Object> value(DebugLookupResultValue(&result));
    Handle<FixedArray> details = Factory::NewFixedArray(2);
    details->set(0, *value);
    details->set(1, result.GetPropertyDetails().AsSmi());
    return *Factory::NewJSArrayWithElements(details);
  }
  return Heap::undefined_value();
}


static Object* Runtime_DebugGetProperty(Arguments args) {
  HandleScope scope;

  ASSERT(args.length() == 2);

  CONVERT_ARG_CHECKED(JSObject, obj, 0);
  CONVERT_ARG_CHECKED(String, name, 1);

  LookupResult result;
  obj->Lookup(*name, &result);
  if (result.IsProperty()) {
    return DebugLookupResultValue(&result);
  }
  return Heap::undefined_value();
}


// Return the names of the local named properties.
// args[0]: object
static Object* Runtime_DebugLocalPropertyNames(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);
  if (!args[0]->IsJSObject()) {
    return Heap::undefined_value();
  }
  CONVERT_ARG_CHECKED(JSObject, obj, 0);

  int n = obj->NumberOfLocalProperties(static_cast<PropertyAttributes>(NONE));
  Handle<FixedArray> names = Factory::NewFixedArray(n);
  obj->GetLocalPropertyNames(*names);
  return *Factory::NewJSArrayWithElements(names);
}


// Return the names of the local indexed properties.
// args[0]: object
static Object* Runtime_DebugLocalElementNames(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);
  if (!args[0]->IsJSObject()) {
    return Heap::undefined_value();
  }
  CONVERT_ARG_CHECKED(JSObject, obj, 0);

  int n = obj->NumberOfLocalElements(static_cast<PropertyAttributes>(NONE));
  Handle<FixedArray> names = Factory::NewFixedArray(n);
  obj->GetLocalElementKeys(*names, static_cast<PropertyAttributes>(NONE));
  return *Factory::NewJSArrayWithElements(names);
}


// Return the property type calculated from the property details.
// args[0]: smi with property details.
static Object* Runtime_DebugPropertyTypeFromDetails(Arguments args) {
  ASSERT(args.length() == 1);
  CONVERT_CHECKED(Smi, details, args[0]);
  PropertyType type = PropertyDetails(details).type();
  return Smi::FromInt(static_cast<int>(type));
}


// Return the property attribute calculated from the property details.
// args[0]: smi with property details.
static Object* Runtime_DebugPropertyAttributesFromDetails(Arguments args) {
  ASSERT(args.length() == 1);
  CONVERT_CHECKED(Smi, details, args[0]);
  PropertyAttributes attributes = PropertyDetails(details).attributes();
  return Smi::FromInt(static_cast<int>(attributes));
}


// Return the property insertion index calculated from the property details.
// args[0]: smi with property details.
static Object* Runtime_DebugPropertyIndexFromDetails(Arguments args) {
  ASSERT(args.length() == 1);
  CONVERT_CHECKED(Smi, details, args[0]);
  int index = PropertyDetails(details).index();
  return Smi::FromInt(index);
}


// Return information on whether an object has a named or indexed interceptor.
// args[0]: object
static Object* Runtime_DebugInterceptorInfo(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);
  if (!args[0]->IsJSObject()) {
    return Smi::FromInt(0);
  }
  CONVERT_ARG_CHECKED(JSObject, obj, 0);

  int result = 0;
  if (obj->HasNamedInterceptor()) result |= 2;
  if (obj->HasIndexedInterceptor()) result |= 1;

  return Smi::FromInt(result);
}


// Return property names from named interceptor.
// args[0]: object
static Object* Runtime_DebugNamedInterceptorPropertyNames(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);
  CONVERT_ARG_CHECKED(JSObject, obj, 0);
  RUNTIME_ASSERT(obj->HasNamedInterceptor());

  v8::Handle<v8::Array> result = GetKeysForNamedInterceptor(obj, obj);
  if (!result.IsEmpty()) return *v8::Utils::OpenHandle(*result);
  return Heap::undefined_value();
}


// Return element names from indexed interceptor.
// args[0]: object
static Object* Runtime_DebugIndexedInterceptorElementNames(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);
  CONVERT_ARG_CHECKED(JSObject, obj, 0);
  RUNTIME_ASSERT(obj->HasIndexedInterceptor());

  v8::Handle<v8::Array> result = GetKeysForIndexedInterceptor(obj, obj);
  if (!result.IsEmpty()) return *v8::Utils::OpenHandle(*result);
  return Heap::undefined_value();
}


// Return property value from named interceptor.
// args[0]: object
// args[1]: property name
static Object* Runtime_DebugNamedInterceptorPropertyValue(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 2);
  CONVERT_ARG_CHECKED(JSObject, obj, 0);
  RUNTIME_ASSERT(obj->HasNamedInterceptor());
  CONVERT_ARG_CHECKED(String, name, 1);

  PropertyAttributes attributes;
  return obj->GetPropertyWithInterceptor(*obj, *name, &attributes);
}


// Return element value from indexed interceptor.
// args[0]: object
// args[1]: index
static Object* Runtime_DebugIndexedInterceptorElementValue(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 2);
  CONVERT_ARG_CHECKED(JSObject, obj, 0);
  RUNTIME_ASSERT(obj->HasIndexedInterceptor());
  CONVERT_NUMBER_CHECKED(uint32_t, index, Uint32, args[1]);

  return obj->GetElementWithInterceptor(*obj, index);
}


static Object* Runtime_CheckExecutionState(Arguments args) {
  ASSERT(args.length() >= 1);
  CONVERT_NUMBER_CHECKED(int, break_id, Int32, args[0]);
  // Check that the break id is valid and that there is a valid frame
  // where execution is broken.
  if (break_id != Top::break_id() ||
      Top::break_frame_id() == StackFrame::NO_ID) {
    return Top::Throw(Heap::illegal_execution_state_symbol());
  }

  return Heap::true_value();
}


static Object* Runtime_GetFrameCount(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);

  // Check arguments.
  Object* result = Runtime_CheckExecutionState(args);
  if (result->IsFailure()) return result;

  // Count all frames which are relevant to debugging stack trace.
  int n = 0;
  StackFrame::Id id = Top::break_frame_id();
  for (JavaScriptFrameIterator it(id); !it.done(); it.Advance()) n++;
  return Smi::FromInt(n);
}


static const int kFrameDetailsFrameIdIndex = 0;
static const int kFrameDetailsReceiverIndex = 1;
static const int kFrameDetailsFunctionIndex = 2;
static const int kFrameDetailsArgumentCountIndex = 3;
static const int kFrameDetailsLocalCountIndex = 4;
static const int kFrameDetailsSourcePositionIndex = 5;
static const int kFrameDetailsConstructCallIndex = 6;
static const int kFrameDetailsDebuggerFrameIndex = 7;
static const int kFrameDetailsFirstDynamicIndex = 8;

// Return an array with frame details
// args[0]: number: break id
// args[1]: number: frame index
//
// The array returned contains the following information:
// 0: Frame id
// 1: Receiver
// 2: Function
// 3: Argument count
// 4: Local count
// 5: Source position
// 6: Constructor call
// 7: Debugger frame
// Arguments name, value
// Locals name, value
static Object* Runtime_GetFrameDetails(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 2);

  // Check arguments.
  Object* check = Runtime_CheckExecutionState(args);
  if (check->IsFailure()) return check;
  CONVERT_NUMBER_CHECKED(int, index, Int32, args[1]);

  // Find the relevant frame with the requested index.
  StackFrame::Id id = Top::break_frame_id();
  int count = 0;
  JavaScriptFrameIterator it(id);
  for (; !it.done(); it.Advance()) {
    if (count == index) break;
    count++;
  }
  if (it.done()) return Heap::undefined_value();

  // Traverse the saved contexts chain to find the active context for the
  // selected frame.
  SaveContext* save = Top::save_context();
  while (save != NULL && !save->below(it.frame())) {
    save = save->prev();
  }
  ASSERT(save != NULL);

  // Get the frame id.
  Handle<Object> frame_id(WrapFrameId(it.frame()->id()));

  // Find source position.
  int position = it.frame()->FindCode()->SourcePosition(it.frame()->pc());

  // Check for constructor frame.
  bool constructor = it.frame()->IsConstructor();

  // Get code and read scope info from it for local variable information.
  Handle<Code> code(it.frame()->FindCode());
  ScopeInfo<> info(*code);

  // Get the context.
  Handle<Context> context(Context::cast(it.frame()->context()));

  // Get the locals names and values into a temporary array.
  //
  // TODO(1240907): Hide compiler-introduced stack variables
  // (e.g. .result)?  For users of the debugger, they will probably be
  // confusing.
  Handle<FixedArray> locals = Factory::NewFixedArray(info.NumberOfLocals() * 2);
  for (int i = 0; i < info.NumberOfLocals(); i++) {
    // Name of the local.
    locals->set(i * 2, *info.LocalName(i));

    // Fetch the value of the local - either from the stack or from a
    // heap-allocated context.
    if (i < info.number_of_stack_slots()) {
      locals->set(i * 2 + 1, it.frame()->GetExpression(i));
    } else {
      Handle<String> name = info.LocalName(i);
      // Traverse the context chain to the function context as all local
      // variables stored in the context will be on the function context.
      while (!context->is_function_context()) {
        context = Handle<Context>(context->previous());
      }
      ASSERT(context->is_function_context());
      locals->set(i * 2 + 1,
                  context->get(ScopeInfo<>::ContextSlotIndex(*code, *name,
                                                             NULL)));
    }
  }

  // Now advance to the arguments adapter frame (if any). If contains all
  // the provided parameters and

  // Now advance to the arguments adapter frame (if any). It contains all
  // the provided parameters whereas the function frame always have the number
  // of arguments matching the functions parameters. The rest of the
  // information (except for what is collected above) is the same.
  it.AdvanceToArgumentsFrame();

  // Find the number of arguments to fill. At least fill the number of
  // parameters for the function and fill more if more parameters are provided.
  int argument_count = info.number_of_parameters();
  if (argument_count < it.frame()->GetProvidedParametersCount()) {
    argument_count = it.frame()->GetProvidedParametersCount();
  }

  // Calculate the size of the result.
  int details_size = kFrameDetailsFirstDynamicIndex +
                     2 * (argument_count + info.NumberOfLocals());
  Handle<FixedArray> details = Factory::NewFixedArray(details_size);

  // Add the frame id.
  details->set(kFrameDetailsFrameIdIndex, *frame_id);

  // Add the function (same as in function frame).
  details->set(kFrameDetailsFunctionIndex, it.frame()->function());

  // Add the arguments count.
  details->set(kFrameDetailsArgumentCountIndex, Smi::FromInt(argument_count));

  // Add the locals count
  details->set(kFrameDetailsLocalCountIndex,
               Smi::FromInt(info.NumberOfLocals()));

  // Add the source position.
  if (position != RelocInfo::kNoPosition) {
    details->set(kFrameDetailsSourcePositionIndex, Smi::FromInt(position));
  } else {
    details->set(kFrameDetailsSourcePositionIndex, Heap::undefined_value());
  }

  // Add the constructor information.
  details->set(kFrameDetailsConstructCallIndex, Heap::ToBoolean(constructor));

  // Add information on whether this frame is invoked in the debugger context.
  details->set(kFrameDetailsDebuggerFrameIndex,
               Heap::ToBoolean(*save->context() == *Debug::debug_context()));

  // Fill the dynamic part.
  int details_index = kFrameDetailsFirstDynamicIndex;

  // Add arguments name and value.
  for (int i = 0; i < argument_count; i++) {
    // Name of the argument.
    if (i < info.number_of_parameters()) {
      details->set(details_index++, *info.parameter_name(i));
    } else {
      details->set(details_index++, Heap::undefined_value());
    }

    // Parameter value.
    if (i < it.frame()->GetProvidedParametersCount()) {
      details->set(details_index++, it.frame()->GetParameter(i));
    } else {
      details->set(details_index++, Heap::undefined_value());
    }
  }

  // Add locals name and value from the temporary copy from the function frame.
  for (int i = 0; i < info.NumberOfLocals() * 2; i++) {
    details->set(details_index++, locals->get(i));
  }

  // Add the receiver (same as in function frame).
  // THIS MUST BE DONE LAST SINCE WE MIGHT ADVANCE
  // THE FRAME ITERATOR TO WRAP THE RECEIVER.
  Handle<Object> receiver(it.frame()->receiver());
  if (!receiver->IsJSObject()) {
    // If the receiver is NOT a JSObject we have hit an optimization
    // where a value object is not converted into a wrapped JS objects.
    // To hide this optimization from the debugger, we wrap the receiver
    // by creating correct wrapper object based on the calling frame's
    // global context.
    it.Advance();
    Handle<Context> calling_frames_global_context(
        Context::cast(Context::cast(it.frame()->context())->global_context()));
    receiver = Factory::ToObject(receiver, calling_frames_global_context);
  }
  details->set(kFrameDetailsReceiverIndex, *receiver);

  ASSERT_EQ(details_size, details_index);
  return *Factory::NewJSArrayWithElements(details);
}


static Object* Runtime_GetCFrames(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);
  Object* result = Runtime_CheckExecutionState(args);
  if (result->IsFailure()) return result;

  static const int kMaxCFramesSize = 200;
  OS::StackFrame frames[kMaxCFramesSize];
  int frames_count = OS::StackWalk(frames, kMaxCFramesSize);
  if (frames_count == OS::kStackWalkError) {
    return Heap::undefined_value();
  }

  Handle<String> address_str = Factory::LookupAsciiSymbol("address");
  Handle<String> text_str = Factory::LookupAsciiSymbol("text");
  Handle<FixedArray> frames_array = Factory::NewFixedArray(frames_count);
  for (int i = 0; i < frames_count; i++) {
    Handle<JSObject> frame_value = Factory::NewJSObject(Top::object_function());
    frame_value->SetProperty(
        *address_str,
        *Factory::NewNumberFromInt(reinterpret_cast<int>(frames[i].address)),
        NONE);

    // Get the stack walk text for this frame.
    Handle<String> frame_text;
    if (strlen(frames[i].text) > 0) {
      Vector<const char> str(frames[i].text, strlen(frames[i].text));
      frame_text = Factory::NewStringFromAscii(str);
    }

    if (!frame_text.is_null()) {
      frame_value->SetProperty(*text_str, *frame_text, NONE);
    }

    frames_array->set(i, *frame_value);
  }
  return *Factory::NewJSArrayWithElements(frames_array);
}


static Object* Runtime_GetBreakLocations(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);

  CONVERT_ARG_CHECKED(JSFunction, raw_fun, 0);
  Handle<SharedFunctionInfo> shared(raw_fun->shared());
  // Find the number of break points
  Handle<Object> break_locations = Debug::GetSourceBreakLocations(shared);
  if (break_locations->IsUndefined()) return Heap::undefined_value();
  // Return array as JS array
  return *Factory::NewJSArrayWithElements(
      Handle<FixedArray>::cast(break_locations));
}


// Set a break point in a function
// args[0]: function
// args[1]: number: break source position (within the function source)
// args[2]: number: break point object
static Object* Runtime_SetFunctionBreakPoint(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 3);
  CONVERT_ARG_CHECKED(JSFunction, raw_fun, 0);
  Handle<SharedFunctionInfo> shared(raw_fun->shared());
  CONVERT_NUMBER_CHECKED(int32_t, source_position, Int32, args[1]);
  RUNTIME_ASSERT(source_position >= 0);
  Handle<Object> break_point_object_arg = args.at<Object>(2);

  // Set break point.
  Debug::SetBreakPoint(shared, source_position, break_point_object_arg);

  return Heap::undefined_value();
}


static Object* FindSharedFunctionInfoInScript(Handle<Script> script,
                                              int position) {
  // Iterate the heap looking for SharedFunctionInfo generated from the
  // script. The inner most SharedFunctionInfo containing the source position
  // for the requested break point is found.
  // NOTE: This might reqire several heap iterations. If the SharedFunctionInfo
  // which is found is not compiled it is compiled and the heap is iterated
  // again as the compilation might create inner functions from the newly
  // compiled function and the actual requested break point might be in one of
  // these functions.
  bool done = false;
  // The current candidate for the source position:
  int target_start_position = RelocInfo::kNoPosition;
  Handle<SharedFunctionInfo> target;
  // The current candidate for the last function in script:
  Handle<SharedFunctionInfo> last;
  while (!done) {
    HeapIterator iterator;
    while (iterator.has_next()) {
      HeapObject* obj = iterator.next();
      ASSERT(obj != NULL);
      if (obj->IsSharedFunctionInfo()) {
        Handle<SharedFunctionInfo> shared(SharedFunctionInfo::cast(obj));
        if (shared->script() == *script) {
          // If the SharedFunctionInfo found has the requested script data and
          // contains the source position it is a candidate.
          int start_position = shared->function_token_position();
          if (start_position == RelocInfo::kNoPosition) {
            start_position = shared->start_position();
          }
          if (start_position <= position &&
              position <= shared->end_position()) {
            // If there is no candidate or this function is within the currrent
            // candidate this is the new candidate.
            if (target.is_null()) {
              target_start_position = start_position;
              target = shared;
            } else {
              if (target_start_position < start_position &&
                  shared->end_position() < target->end_position()) {
                target_start_position = start_position;
                target = shared;
              }
            }
          }

          // Keep track of the last function in the script.
          if (last.is_null() ||
              shared->end_position() > last->start_position()) {
            last = shared;
          }
        }
      }
    }

    // Make sure some candidate is selected.
    if (target.is_null()) {
      if (!last.is_null()) {
        // Position after the last function - use last.
        target = last;
      } else {
        // Unable to find function - possibly script without any function.
        return Heap::undefined_value();
      }
    }

    // If the candidate found is compiled we are done. NOTE: when lazy
    // compilation of inner functions is introduced some additional checking
    // needs to be done here to compile inner functions.
    done = target->is_compiled();
    if (!done) {
      // If the candidate is not compiled compile it to reveal any inner
      // functions which might contain the requested source position.
      CompileLazyShared(target, KEEP_EXCEPTION, 0);
    }
  }

  return *target;
}


// Change the state of a break point in a script. NOTE: Regarding performance
// see the NOTE for GetScriptFromScriptData.
// args[0]: script to set break point in
// args[1]: number: break source position (within the script source)
// args[2]: number: break point object
static Object* Runtime_SetScriptBreakPoint(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 3);
  CONVERT_ARG_CHECKED(JSValue, wrapper, 0);
  CONVERT_NUMBER_CHECKED(int32_t, source_position, Int32, args[1]);
  RUNTIME_ASSERT(source_position >= 0);
  Handle<Object> break_point_object_arg = args.at<Object>(2);

  // Get the script from the script wrapper.
  RUNTIME_ASSERT(wrapper->value()->IsScript());
  Handle<Script> script(Script::cast(wrapper->value()));

  Object* result = FindSharedFunctionInfoInScript(script, source_position);
  if (!result->IsUndefined()) {
    Handle<SharedFunctionInfo> shared(SharedFunctionInfo::cast(result));
    // Find position within function. The script position might be before the
    // source position of the first function.
    int position;
    if (shared->start_position() > source_position) {
      position = 0;
    } else {
      position = source_position - shared->start_position();
    }
    Debug::SetBreakPoint(shared, position, break_point_object_arg);
  }
  return  Heap::undefined_value();
}


// Clear a break point
// args[0]: number: break point object
static Object* Runtime_ClearBreakPoint(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 1);
  Handle<Object> break_point_object_arg = args.at<Object>(0);

  // Clear break point.
  Debug::ClearBreakPoint(break_point_object_arg);

  return Heap::undefined_value();
}


// Change the state of break on exceptions
// args[0]: boolean indicating uncaught exceptions
// args[1]: boolean indicating on/off
static Object* Runtime_ChangeBreakOnException(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 2);
  ASSERT(args[0]->IsNumber());
  ASSERT(args[1]->IsBoolean());

  // Update break point state
  ExceptionBreakType type =
      static_cast<ExceptionBreakType>(NumberToUint32(args[0]));
  bool enable = args[1]->ToBoolean()->IsTrue();
  Debug::ChangeBreakOnException(type, enable);
  return Heap::undefined_value();
}


// Prepare for stepping
// args[0]: break id for checking execution state
// args[1]: step action from the enumeration StepAction
// args[2]: number of times to perform the step
static Object* Runtime_PrepareStep(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 3);
  // Check arguments.
  Object* check = Runtime_CheckExecutionState(args);
  if (check->IsFailure()) return check;
  if (!args[1]->IsNumber() || !args[2]->IsNumber()) {
    return Top::Throw(Heap::illegal_argument_symbol());
  }

  // Get the step action and check validity.
  StepAction step_action = static_cast<StepAction>(NumberToInt32(args[1]));
  if (step_action != StepIn &&
      step_action != StepNext &&
      step_action != StepOut &&
      step_action != StepInMin &&
      step_action != StepMin) {
    return Top::Throw(Heap::illegal_argument_symbol());
  }

  // Get the number of steps.
  int step_count = NumberToInt32(args[2]);
  if (step_count < 1) {
    return Top::Throw(Heap::illegal_argument_symbol());
  }

  // Prepare step.
  Debug::PrepareStep(static_cast<StepAction>(step_action), step_count);
  return Heap::undefined_value();
}


// Clear all stepping set by PrepareStep.
static Object* Runtime_ClearStepping(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 0);
  Debug::ClearStepping();
  return Heap::undefined_value();
}


// Creates a copy of the with context chain. The copy of the context chain is
// is linked to the function context supplied.
static Handle<Context> CopyWithContextChain(Handle<Context> context_chain,
                                            Handle<Context> function_context) {
  // At the bottom of the chain. Return the function context to link to.
  if (context_chain->is_function_context()) {
    return function_context;
  }

  // Recursively copy the with contexts.
  Handle<Context> previous(context_chain->previous());
  Handle<JSObject> extension(JSObject::cast(context_chain->extension()));
  return Factory::NewWithContext(
      CopyWithContextChain(function_context, previous), extension);
}


// Helper function to find or create the arguments object for
// Runtime_DebugEvaluate.
static Handle<Object> GetArgumentsObject(JavaScriptFrame* frame,
                                         Handle<JSFunction> function,
                                         Handle<Code> code,
                                         const ScopeInfo<>* sinfo,
                                         Handle<Context> function_context) {
  // Try to find the value of 'arguments' to pass as parameter. If it is not
  // found (that is the debugged function does not reference 'arguments' and
  // does not support eval) then create an 'arguments' object.
  int index;
  if (sinfo->number_of_stack_slots() > 0) {
    index = ScopeInfo<>::StackSlotIndex(*code, Heap::arguments_symbol());
    if (index != -1) {
      return Handle<Object>(frame->GetExpression(index));
    }
  }

  if (sinfo->number_of_context_slots() > Context::MIN_CONTEXT_SLOTS) {
    index = ScopeInfo<>::ContextSlotIndex(*code, Heap::arguments_symbol(),
                                          NULL);
    if (index != -1) {
      return Handle<Object>(function_context->get(index));
    }
  }

  const int length = frame->GetProvidedParametersCount();
  Handle<JSObject> arguments = Factory::NewArgumentsObject(function, length);
  Handle<FixedArray> array = Factory::NewFixedArray(length);
  WriteBarrierMode mode = array->GetWriteBarrierMode();
  for (int i = 0; i < length; i++) {
    array->set(i, frame->GetParameter(i), mode);
  }
  arguments->set_elements(*array);
  return arguments;
}


// Evaluate a piece of JavaScript in the context of a stack frame for
// debugging. This is acomplished by creating a new context which in its
// extension part has all the parameters and locals of the function on the
// stack frame. A function which calls eval with the code to evaluate is then
// compiled in this context and called in this context. As this context
// replaces the context of the function on the stack frame a new (empty)
// function is created as well to be used as the closure for the context.
// This function and the context acts as replacements for the function on the
// stack frame presenting the same view of the values of parameters and
// local variables as if the piece of JavaScript was evaluated at the point
// where the function on the stack frame is currently stopped.
static Object* Runtime_DebugEvaluate(Arguments args) {
  HandleScope scope;

  // Check the execution state and decode arguments frame and source to be
  // evaluated.
  ASSERT(args.length() == 4);
  Object* check_result = Runtime_CheckExecutionState(args);
  if (check_result->IsFailure()) return check_result;
  CONVERT_CHECKED(Smi, wrapped_id, args[1]);
  CONVERT_ARG_CHECKED(String, source, 2);
  CONVERT_BOOLEAN_CHECKED(disable_break, args[3]);

  // Handle the processing of break.
  DisableBreak disable_break_save(disable_break);

  // Get the frame where the debugging is performed.
  StackFrame::Id id = UnwrapFrameId(wrapped_id);
  JavaScriptFrameIterator it(id);
  JavaScriptFrame* frame = it.frame();
  Handle<JSFunction> function(JSFunction::cast(frame->function()));
  Handle<Code> code(function->code());
  ScopeInfo<> sinfo(*code);

  // Traverse the saved contexts chain to find the active context for the
  // selected frame.
  SaveContext* save = Top::save_context();
  while (save != NULL && !save->below(frame)) {
    save = save->prev();
  }
  ASSERT(save != NULL);
  SaveContext savex;
  Top::set_context(*(save->context()));

  // Create the (empty) function replacing the function on the stack frame for
  // the purpose of evaluating in the context created below. It is important
  // that this function does not describe any parameters and local variables
  // in the context. If it does then this will cause problems with the lookup
  // in Context::Lookup, where context slots for parameters and local variables
  // are looked at before the extension object.
  Handle<JSFunction> go_between =
      Factory::NewFunction(Factory::empty_string(), Factory::undefined_value());
  go_between->set_context(function->context());
#ifdef DEBUG
  ScopeInfo<> go_between_sinfo(go_between->shared()->code());
  ASSERT(go_between_sinfo.number_of_parameters() == 0);
  ASSERT(go_between_sinfo.number_of_context_slots() == 0);
#endif

  // Allocate and initialize a context extension object with all the
  // arguments, stack locals heap locals and extension properties of the
  // debugged function.
  Handle<JSObject> context_ext = Factory::NewJSObject(Top::object_function());
  // First fill all parameters to the context extension.
  for (int i = 0; i < sinfo.number_of_parameters(); ++i) {
    SetProperty(context_ext,
                sinfo.parameter_name(i),
                Handle<Object>(frame->GetParameter(i)), NONE);
  }
  // Second fill all stack locals to the context extension.
  for (int i = 0; i < sinfo.number_of_stack_slots(); i++) {
    SetProperty(context_ext,
                sinfo.stack_slot_name(i),
                Handle<Object>(frame->GetExpression(i)), NONE);
  }
  // Third fill all context locals to the context extension.
  Handle<Context> frame_context(Context::cast(frame->context()));
  Handle<Context> function_context(frame_context->fcontext());
  for (int i = Context::MIN_CONTEXT_SLOTS;
       i < sinfo.number_of_context_slots();
       ++i) {
    int context_index =
        ScopeInfo<>::ContextSlotIndex(*code, *sinfo.context_slot_name(i), NULL);
    SetProperty(context_ext,
                sinfo.context_slot_name(i),
                Handle<Object>(function_context->get(context_index)), NONE);
  }
  // Finally copy any properties from the function context extension. This will
  // be variables introduced by eval.
  if (function_context->has_extension() &&
      !function_context->IsGlobalContext()) {
    Handle<JSObject> ext(JSObject::cast(function_context->extension()));
    Handle<FixedArray> keys = GetKeysInFixedArrayFor(ext);
    for (int i = 0; i < keys->length(); i++) {
      // Names of variables introduced by eval are strings.
      ASSERT(keys->get(i)->IsString());
      Handle<String> key(String::cast(keys->get(i)));
      SetProperty(context_ext, key, GetProperty(ext, key), NONE);
    }
  }

  // Allocate a new context for the debug evaluation and set the extension
  // object build.
  Handle<Context> context =
      Factory::NewFunctionContext(Context::MIN_CONTEXT_SLOTS, go_between);
  context->set_extension(*context_ext);
  // Copy any with contexts present and chain them in front of this context.
  context = CopyWithContextChain(frame_context, context);

  // Wrap the evaluation statement in a new function compiled in the newly
  // created context. The function has one parameter which has to be called
  // 'arguments'. This it to have access to what would have been 'arguments' in
  // the function beeing debugged.
  // function(arguments,__source__) {return eval(__source__);}
  static const char* source_str =
      "function(arguments,__source__){return eval(__source__);}";
  static const int source_str_length = strlen(source_str);
  Handle<String> function_source =
      Factory::NewStringFromAscii(Vector<const char>(source_str,
                                                     source_str_length));
  Handle<JSFunction> boilerplate =
      Compiler::CompileEval(function_source, 0, context->IsGlobalContext());
  if (boilerplate.is_null()) return Failure::Exception();
  Handle<JSFunction> compiled_function =
      Factory::NewFunctionFromBoilerplate(boilerplate, context);

  // Invoke the result of the compilation to get the evaluation function.
  bool has_pending_exception;
  Handle<Object> receiver(frame->receiver());
  Handle<Object> evaluation_function =
      Execution::Call(compiled_function, receiver, 0, NULL,
                      &has_pending_exception);
  if (has_pending_exception) return Failure::Exception();

  Handle<Object> arguments = GetArgumentsObject(frame, function, code, &sinfo,
                                                function_context);

  // Invoke the evaluation function and return the result.
  const int argc = 2;
  Object** argv[argc] = { arguments.location(),
                          Handle<Object>::cast(source).location() };
  Handle<Object> result =
      Execution::Call(Handle<JSFunction>::cast(evaluation_function), receiver,
                      argc, argv, &has_pending_exception);
  if (has_pending_exception) return Failure::Exception();
  return *result;
}


static Object* Runtime_DebugEvaluateGlobal(Arguments args) {
  HandleScope scope;

  // Check the execution state and decode arguments frame and source to be
  // evaluated.
  ASSERT(args.length() == 3);
  Object* check_result = Runtime_CheckExecutionState(args);
  if (check_result->IsFailure()) return check_result;
  CONVERT_ARG_CHECKED(String, source, 1);
  CONVERT_BOOLEAN_CHECKED(disable_break, args[2]);

  // Handle the processing of break.
  DisableBreak disable_break_save(disable_break);

  // Enter the top context from before the debugger was invoked.
  SaveContext save;
  SaveContext* top = &save;
  while (top != NULL && *top->context() == *Debug::debug_context()) {
    top = top->prev();
  }
  if (top != NULL) {
    Top::set_context(*top->context());
  }

  // Get the global context now set to the top context from before the
  // debugger was invoked.
  Handle<Context> context = Top::global_context();

  // Compile the source to be evaluated.
  Handle<JSFunction> boilerplate(Compiler::CompileEval(source, 0, true));
  if (boilerplate.is_null()) return Failure::Exception();
  Handle<JSFunction> compiled_function =
      Handle<JSFunction>(Factory::NewFunctionFromBoilerplate(boilerplate,
                                                             context));

  // Invoke the result of the compilation to get the evaluation function.
  bool has_pending_exception;
  Handle<Object> receiver = Top::global();
  Handle<Object> result =
    Execution::Call(compiled_function, receiver, 0, NULL,
                    &has_pending_exception);
  if (has_pending_exception) return Failure::Exception();
  return *result;
}


// Helper function used by Runtime_DebugGetLoadedScripts below.
static int DebugGetLoadedScripts(FixedArray* instances, int instances_size) {
  NoHandleAllocation ha;
  AssertNoAllocation no_alloc;

  // Get hold of the current empty script.
  Context* context = Top::context()->global_context();
  Script* empty = context->empty_script();

  // Scan heap for Script objects.
  int count = 0;
  HeapIterator iterator;
  while (iterator.has_next()) {
    HeapObject* obj = iterator.next();
    ASSERT(obj != NULL);
    if (obj->IsScript() && obj != empty) {
      if (instances != NULL && count < instances_size) {
        instances->set(count, obj);
      }
      count++;
    }
  }

  return count;
}


static Object* Runtime_DebugGetLoadedScripts(Arguments args) {
  HandleScope scope;
  ASSERT(args.length() == 0);

  // Perform two GCs to get rid of all unreferenced scripts. The first GC gets
  // rid of all the cached script wrappes and the second gets rid of the
  // scripts which is no longer referenced.
  Heap::CollectAllGarbage();
  Heap::CollectAllGarbage();

  // Get the number of scripts.
  int count;
  count = DebugGetLoadedScripts(NULL, 0);

  // Allocate an array to hold the result.
  Handle<FixedArray> instances = Factory::NewFixedArray(count);

  // Fill the script objects.
  count = DebugGetLoadedScripts(*instances, count);

  // Convert the script objects to proper JS objects.
  for (int i = 0; i < count; i++) {
    Handle<Script> script = Handle<Script>(Script::cast(instances->get(i)));
    // Get the script wrapper in a local handle before calling GetScriptWrapper,
    // because using
    //   instances->set(i, *GetScriptWrapper(script))
    // is unsafe as GetScriptWrapper might call GC and the C++ compiler might
    // already have deferenced the instances handle.
    Handle<JSValue> wrapper = GetScriptWrapper(script);
    instances->set(i, *wrapper);
  }

  // Return result as a JS array.
  Handle<JSObject> result = Factory::NewJSObject(Top::array_function());
  Handle<JSArray>::cast(result)->SetContent(*instances);
  return *result;
}


// Helper function used by Runtime_DebugReferencedBy below.
static int DebugReferencedBy(JSObject* target,
                             Object* instance_filter, int max_references,
                             FixedArray* instances, int instances_size,
                             JSFunction* context_extension_function,
                             JSFunction* arguments_function) {
  NoHandleAllocation ha;
  AssertNoAllocation no_alloc;

  // Iterate the heap.
  int count = 0;
  JSObject* last = NULL;
  HeapIterator iterator;
  while (iterator.has_next() &&
         (max_references == 0 || count < max_references)) {
    // Only look at all JSObjects.
    HeapObject* heap_obj = iterator.next();
    if (heap_obj->IsJSObject()) {
      // Skip context extension objects and argument arrays as these are
      // checked in the context of functions using them.
      JSObject* obj = JSObject::cast(heap_obj);
      if (obj->map()->constructor() == context_extension_function ||
          obj->map()->constructor() == arguments_function) {
        continue;
      }

      // Check if the JS object has a reference to the object looked for.
      if (obj->ReferencesObject(target)) {
        // Check instance filter if supplied. This is normally used to avoid
        // references from mirror objects (see Runtime_IsInPrototypeChain).
        if (!instance_filter->IsUndefined()) {
          Object* V = obj;
          while (true) {
            Object* prototype = V->GetPrototype();
            if (prototype->IsNull()) {
              break;
            }
            if (instance_filter == prototype) {
              obj = NULL;  // Don't add this object.
              break;
            }
            V = prototype;
          }
        }

        if (obj != NULL) {
          // Valid reference found add to instance array if supplied an update
          // count.
          if (instances != NULL && count < instances_size) {
            instances->set(count, obj);
          }
          last = obj;
          count++;
        }
      }
    }
  }

  // Check for circular reference only. This can happen when the object is only
  // referenced from mirrors and has a circular reference in which case the
  // object is not really alive and would have been garbage collected if not
  // referenced from the mirror.
  if (count == 1 && last == target) {
    count = 0;
  }

  // Return the number of referencing objects found.
  return count;
}


// Scan the heap for objects with direct references to an object
// args[0]: the object to find references to
// args[1]: constructor function for instances to exclude (Mirror)
// args[2]: the the maximum number of objects to return
static Object* Runtime_DebugReferencedBy(Arguments args) {
  ASSERT(args.length() == 3);

  // First perform a full GC in order to avoid references from dead objects.
  Heap::CollectAllGarbage();

  // Check parameters.
  CONVERT_CHECKED(JSObject, target, args[0]);
  Object* instance_filter = args[1];
  RUNTIME_ASSERT(instance_filter->IsUndefined() ||
                 instance_filter->IsJSObject());
  CONVERT_NUMBER_CHECKED(int32_t, max_references, Int32, args[2]);
  RUNTIME_ASSERT(max_references >= 0);

  // Get the constructor function for context extension and arguments array.
  JSFunction* context_extension_function =
      Top::context()->global_context()->context_extension_function();
  JSObject* arguments_boilerplate =
      Top::context()->global_context()->arguments_boilerplate();
  JSFunction* arguments_function =
      JSFunction::cast(arguments_boilerplate->map()->constructor());

  // Get the number of referencing objects.
  int count;
  count = DebugReferencedBy(target, instance_filter, max_references,
                            NULL, 0,
                            context_extension_function, arguments_function);

  // Allocate an array to hold the result.
  Object* object = Heap::AllocateFixedArray(count);
  if (object->IsFailure()) return object;
  FixedArray* instances = FixedArray::cast(object);

  // Fill the referencing objects.
  count = DebugReferencedBy(target, instance_filter, max_references,
                            instances, count,
                            context_extension_function, arguments_function);

  // Return result as JS array.
  Object* result =
      Heap::AllocateJSObject(
          Top::context()->global_context()->array_function());
  if (!result->IsFailure()) JSArray::cast(result)->SetContent(instances);
  return result;
}


// Helper function used by Runtime_DebugConstructedBy below.
static int DebugConstructedBy(JSFunction* constructor, int max_references,
                              FixedArray* instances, int instances_size) {
  AssertNoAllocation no_alloc;

  // Iterate the heap.
  int count = 0;
  HeapIterator iterator;
  while (iterator.has_next() &&
         (max_references == 0 || count < max_references)) {
    // Only look at all JSObjects.
    HeapObject* heap_obj = iterator.next();
    if (heap_obj->IsJSObject()) {
      JSObject* obj = JSObject::cast(heap_obj);
      if (obj->map()->constructor() == constructor) {
        // Valid reference found add to instance array if supplied an update
        // count.
        if (instances != NULL && count < instances_size) {
          instances->set(count, obj);
        }
        count++;
      }
    }
  }

  // Return the number of referencing objects found.
  return count;
}


// Scan the heap for objects constructed by a specific function.
// args[0]: the constructor to find instances of
// args[1]: the the maximum number of objects to return
static Object* Runtime_DebugConstructedBy(Arguments args) {
  ASSERT(args.length() == 2);

  // First perform a full GC in order to avoid dead objects.
  Heap::CollectAllGarbage();

  // Check parameters.
  CONVERT_CHECKED(JSFunction, constructor, args[0]);
  CONVERT_NUMBER_CHECKED(int32_t, max_references, Int32, args[1]);
  RUNTIME_ASSERT(max_references >= 0);

  // Get the number of referencing objects.
  int count;
  count = DebugConstructedBy(constructor, max_references, NULL, 0);

  // Allocate an array to hold the result.
  Object* object = Heap::AllocateFixedArray(count);
  if (object->IsFailure()) return object;
  FixedArray* instances = FixedArray::cast(object);

  // Fill the referencing objects.
  count = DebugConstructedBy(constructor, max_references, instances, count);

  // Return result as JS array.
  Object* result =
      Heap::AllocateJSObject(
          Top::context()->global_context()->array_function());
  if (!result->IsFailure()) JSArray::cast(result)->SetContent(instances);
  return result;
}


static Object* Runtime_GetPrototype(Arguments args) {
  ASSERT(args.length() == 1);

  CONVERT_CHECKED(JSObject, obj, args[0]);

  return obj->GetPrototype();
}


static Object* Runtime_SystemBreak(Arguments args) {
  ASSERT(args.length() == 0);
  CPU::DebugBreak();
  return Heap::undefined_value();
}


// Finds the script object from the script data. NOTE: This operation uses
// heap traversal to find the function generated for the source position
// for the requested break point. For lazily compiled functions several heap
// traversals might be required rendering this operation as a rather slow
// operation. However for setting break points which is normally done through
// some kind of user interaction the performance is not crucial.
static Handle<Object> Runtime_GetScriptFromScriptName(
    Handle<String> script_name) {
  // Scan the heap for Script objects to find the script with the requested
  // script data.
  Handle<Script> script;
  HeapIterator iterator;
  while (script.is_null() && iterator.has_next()) {
    HeapObject* obj = iterator.next();
    // If a script is found check if it has the script data requested.
    if (obj->IsScript()) {
      if (Script::cast(obj)->name()->IsString()) {
        if (String::cast(Script::cast(obj)->name())->Equals(*script_name)) {
          script = Handle<Script>(Script::cast(obj));
        }
      }
    }
  }

  // If no script with the requested script data is found return undefined.
  if (script.is_null()) return Factory::undefined_value();

  // Return the script found.
  return GetScriptWrapper(script);
}


// Get the script object from script data. NOTE: Regarding performance
// see the NOTE for GetScriptFromScriptData.
// args[0]: script data for the script to find the source for
static Object* Runtime_GetScript(Arguments args) {
  HandleScope scope;

  ASSERT(args.length() == 1);

  CONVERT_CHECKED(String, script_name, args[0]);

  // Find the requested script.
  Handle<Object> result =
      Runtime_GetScriptFromScriptName(Handle<String>(script_name));
  return *result;
}


static Object* Runtime_FunctionGetAssemblerCode(Arguments args) {
#ifdef DEBUG
  HandleScope scope;
  ASSERT(args.length() == 1);
  // Get the function and make sure it is compiled.
  CONVERT_ARG_CHECKED(JSFunction, func, 0);
  if (!func->is_compiled() && !CompileLazy(func, KEEP_EXCEPTION)) {
    return Failure::Exception();
  }
  func->code()->PrintLn();
#endif  // DEBUG
  return Heap::undefined_value();
}


static Object* Runtime_Abort(Arguments args) {
  ASSERT(args.length() == 2);
  OS::PrintError("abort: %s\n", reinterpret_cast<char*>(args[0]) +
                                    Smi::cast(args[1])->value());
  Top::PrintStack();
  OS::Abort();
  UNREACHABLE();
  return NULL;
}


#ifdef DEBUG
// ListNatives is ONLY used by the fuzz-natives.js in debug mode
// Exclude the code in release mode.
static Object* Runtime_ListNatives(Arguments args) {
  ASSERT(args.length() == 0);
  HandleScope scope;
  Handle<JSArray> result = Factory::NewJSArray(0);
  int index = 0;
#define ADD_ENTRY(Name, argc)                                                \
  {                                                                          \
    HandleScope inner;                                                       \
    Handle<String> name =                                                    \
      Factory::NewStringFromAscii(Vector<const char>(#Name, strlen(#Name))); \
    Handle<JSArray> pair = Factory::NewJSArray(0);                           \
    SetElement(pair, 0, name);                                               \
    SetElement(pair, 1, Handle<Smi>(Smi::FromInt(argc)));                    \
    SetElement(result, index++, pair);                                       \
  }
  RUNTIME_FUNCTION_LIST(ADD_ENTRY)
#undef ADD_ENTRY
  return *result;
}
#endif


static Object* Runtime_IS_VAR(Arguments args) {
  UNREACHABLE();  // implemented as macro in the parser
  return NULL;
}


// ----------------------------------------------------------------------------
// Implementation of Runtime

