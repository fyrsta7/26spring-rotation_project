static bool LookupForWrite(Handle<JSObject> receiver,
                           Handle<String> name,
                           LookupResult* lookup) {
  receiver->LocalLookup(*name, lookup);
  if (!StoreICableLookup(lookup)) {
    // 2nd chance: There can be accessors somewhere in the prototype chain. Note
    // that we explicitly exclude native accessors for now, because the stubs
    // are not yet prepared for this scenario.
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
