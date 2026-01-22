static bool LookupForWrite(Handle<JSObject> receiver,
                           Handle<String> name,
                           LookupResult* lookup) {
  receiver->LocalLookup(*name, lookup);
  if (!StoreICableLookup(lookup)) {
    // 2nd chance: There can be accessors somewhere in the prototype chain, but
    // for compatibility reasons we have to hide this behind a flag. Note that
    // we explicitly exclude native accessors for now, because the stubs are not
    // yet prepared for this scenario.
    if (!FLAG_es5_readonly) return false;
    receiver->Lookup(*name, lookup);
    if (!lookup->IsCallbacks()) return false;
    Handle<Object> callback(lookup->GetCallbackObject());
    return callback->IsAccessorPair() && StoreICableLookup(lookup);
  }

  if (lookup->IsInterceptor() &&
      receiver->GetNamedInterceptor()->setter()->IsUndefined()) {
    receiver->LocalLookupRealNamedProperty(*name, lookup);
    return StoreICableLookup(lookup);
  }

  return true;
}
