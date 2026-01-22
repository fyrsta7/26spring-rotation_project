void LookupIterator::InternalUpdateProtector(Isolate* isolate,
                                             Handle<Object> receiver_generic,
                                             Handle<Name> name) {
  if (isolate->bootstrapper()->IsActive()) return;
  if (!receiver_generic->IsHeapObject()) return;
  Handle<HeapObject> receiver = Handle<HeapObject>::cast(receiver_generic);

  ReadOnlyRoots roots(isolate);
  if (*name == roots.constructor_string()) {
    // Setting the constructor property could change an instance's @@species
    if (receiver->IsJSArray(isolate)) {
      if (!Protectors::IsArraySpeciesLookupChainIntact(isolate)) return;
      isolate->CountUsage(
          v8::Isolate::UseCounterFeature::kArrayInstanceConstructorModified);
      Protectors::InvalidateArraySpeciesLookupChain(isolate);
      return;
    } else if (receiver->IsJSPromise(isolate)) {
      if (!Protectors::IsPromiseSpeciesLookupChainIntact(isolate)) return;
      Protectors::InvalidatePromiseSpeciesLookupChain(isolate);
      return;
    } else if (receiver->IsJSRegExp(isolate)) {
      if (!Protectors::IsRegExpSpeciesLookupChainIntact(isolate)) return;
      Protectors::InvalidateRegExpSpeciesLookupChain(isolate);
      return;
    } else if (receiver->IsJSTypedArray(isolate)) {
      if (!Protectors::IsTypedArraySpeciesLookupChainIntact(isolate)) return;
      Protectors::InvalidateTypedArraySpeciesLookupChain(isolate);
      return;
    }
    if (receiver->map(isolate).is_prototype_map()) {
      DisallowGarbageCollection no_gc;
      // Setting the constructor of any prototype with the @@species protector
      // (of any realm) also needs to invalidate the protector.
      if (isolate->IsInAnyContext(*receiver,
                                  Context::INITIAL_ARRAY_PROTOTYPE_INDEX)) {
        if (!Protectors::IsArraySpeciesLookupChainIntact(isolate)) return;
        isolate->CountUsage(
            v8::Isolate::UseCounterFeature::kArrayPrototypeConstructorModified);
        Protectors::InvalidateArraySpeciesLookupChain(isolate);
      } else if (receiver->IsJSPromisePrototype()) {
        if (!Protectors::IsPromiseSpeciesLookupChainIntact(isolate)) return;
        Protectors::InvalidatePromiseSpeciesLookupChain(isolate);
      } else if (receiver->IsJSRegExpPrototype()) {
        if (!Protectors::IsRegExpSpeciesLookupChainIntact(isolate)) return;
        Protectors::InvalidateRegExpSpeciesLookupChain(isolate);
      } else if (receiver->IsJSTypedArrayPrototype()) {
        if (!Protectors::IsTypedArraySpeciesLookupChainIntact(isolate)) return;
        Protectors::InvalidateTypedArraySpeciesLookupChain(isolate);
      }
    }
  } else if (*name == roots.next_string()) {
    if (receiver->IsJSArrayIterator() ||
        receiver->IsJSArrayIteratorPrototype()) {
      // Setting the next property of %ArrayIteratorPrototype% also needs to
      // invalidate the array iterator protector.
      if (!Protectors::IsArrayIteratorLookupChainIntact(isolate)) return;
      Protectors::InvalidateArrayIteratorLookupChain(isolate);
    } else if (receiver->IsJSMapIterator() ||
               receiver->IsJSMapIteratorPrototype()) {
      if (!Protectors::IsMapIteratorLookupChainIntact(isolate)) return;
      Protectors::InvalidateMapIteratorLookupChain(isolate);
    } else if (receiver->IsJSSetIterator() ||
               receiver->IsJSSetIteratorPrototype()) {
      if (!Protectors::IsSetIteratorLookupChainIntact(isolate)) return;
      Protectors::InvalidateSetIteratorLookupChain(isolate);
    } else if (receiver->IsJSStringIterator() ||
               receiver->IsJSStringIteratorPrototype()) {
      // Setting the next property of %StringIteratorPrototype% invalidates the
      // string iterator protector.
      if (!Protectors::IsStringIteratorLookupChainIntact(isolate)) return;
      Protectors::InvalidateStringIteratorLookupChain(isolate);
    }
  } else if (*name == roots.species_symbol()) {
    // Setting the Symbol.species property of any Array, Promise or TypedArray
    // constructor invalidates the @@species protector
    if (receiver->IsJSArrayConstructor()) {
      if (!Protectors::IsArraySpeciesLookupChainIntact(isolate)) return;
      isolate->CountUsage(
          v8::Isolate::UseCounterFeature::kArraySpeciesModified);
      Protectors::InvalidateArraySpeciesLookupChain(isolate);
    } else if (receiver->IsJSPromiseConstructor()) {
      if (!Protectors::IsPromiseSpeciesLookupChainIntact(isolate)) return;
      Protectors::InvalidatePromiseSpeciesLookupChain(isolate);
    } else if (receiver->IsJSRegExpConstructor()) {
      if (!Protectors::IsRegExpSpeciesLookupChainIntact(isolate)) return;
      Protectors::InvalidateRegExpSpeciesLookupChain(isolate);
    } else if (receiver->IsTypedArrayConstructor()) {
      if (!Protectors::IsTypedArraySpeciesLookupChainIntact(isolate)) return;
      Protectors::InvalidateTypedArraySpeciesLookupChain(isolate);
    }
  } else if (*name == roots.is_concat_spreadable_symbol()) {
    if (!Protectors::IsIsConcatSpreadableLookupChainIntact(isolate)) return;
    Protectors::InvalidateIsConcatSpreadableLookupChain(isolate);
  } else if (*name == roots.iterator_symbol()) {
    if (receiver->IsJSArray(isolate)) {
      if (!Protectors::IsArrayIteratorLookupChainIntact(isolate)) return;
      Protectors::InvalidateArrayIteratorLookupChain(isolate);
    } else if (receiver->IsJSSet(isolate) || receiver->IsJSSetIterator() ||
               receiver->IsJSSetIteratorPrototype() ||
               receiver->IsJSSetPrototype()) {
      if (Protectors::IsSetIteratorLookupChainIntact(isolate)) {
        Protectors::InvalidateSetIteratorLookupChain(isolate);
      }
    } else if (receiver->IsJSMapIterator() ||
               receiver->IsJSMapIteratorPrototype()) {
      if (Protectors::IsMapIteratorLookupChainIntact(isolate)) {
        Protectors::InvalidateMapIteratorLookupChain(isolate);
      }
    } else if (receiver->IsJSIteratorPrototype()) {
      if (Protectors::IsMapIteratorLookupChainIntact(isolate)) {
        Protectors::InvalidateMapIteratorLookupChain(isolate);
      }
      if (Protectors::IsSetIteratorLookupChainIntact(isolate)) {
        Protectors::InvalidateSetIteratorLookupChain(isolate);
      }
    } else if (isolate->IsInAnyContext(
                   *receiver, Context::INITIAL_STRING_PROTOTYPE_INDEX)) {
      // Setting the Symbol.iterator property of String.prototype invalidates
      // the string iterator protector. Symbol.iterator can also be set on a
      // String wrapper, but not on a primitive string. We only support
      // protector for primitive strings.
      if (!Protectors::IsStringIteratorLookupChainIntact(isolate)) return;
      Protectors::InvalidateStringIteratorLookupChain(isolate);
    }
  } else if (*name == roots.resolve_string()) {
    if (!Protectors::IsPromiseResolveLookupChainIntact(isolate)) return;
    // Setting the "resolve" property on any %Promise% intrinsic object
    // invalidates the Promise.resolve protector.
    if (receiver->IsJSPromiseConstructor()) {
      Protectors::InvalidatePromiseResolveLookupChain(isolate);
    }
  } else if (*name == roots.then_string()) {
    if (!Protectors::IsPromiseThenLookupChainIntact(isolate)) return;
    // Setting the "then" property on any JSPromise instance or on the
    // initial %PromisePrototype% invalidates the Promise#then protector.
    // Also setting the "then" property on the initial %ObjectPrototype%
    // invalidates the Promise#then protector, since we use this protector
    // to guard the fast-path in AsyncGeneratorResolve, where we can skip
    // the ResolvePromise step and go directly to FulfillPromise if we
    // know that the Object.prototype doesn't contain a "then" method.
    if (receiver->IsJSPromise(isolate) || receiver->IsJSObjectPrototype() ||
        receiver->IsJSPromisePrototype()) {
      Protectors::InvalidatePromiseThenLookupChain(isolate);
    }
  }
}
