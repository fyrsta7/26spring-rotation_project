void LookupIterator::TransitionToAccessorProperty(
    AccessorComponent component, Handle<Object> accessor,
    PropertyAttributes attributes) {
  DCHECK(!accessor->IsNull());
  // Can only be called when the receiver is a JSObject. JSProxy has to be
  // handled via a trap. Adding properties to primitive values is not
  // observable.
  Handle<JSObject> receiver = GetStoreTarget();

  if (!IsElement() && !receiver->map()->is_dictionary_map()) {
    holder_ = receiver;
    Handle<Map> old_map(receiver->map(), isolate_);
    Handle<Map> new_map = Map::TransitionToAccessorProperty(
        old_map, name_, component, accessor, attributes);
    bool simple_transition = new_map->GetBackPointer() == receiver->map();
    JSObject::MigrateToMap(receiver, new_map);

    if (simple_transition) {
      int number = new_map->LastAdded();
      number_ = static_cast<uint32_t>(number);
      property_details_ = new_map->GetLastDescriptorDetails();
      state_ = ACCESSOR;
      return;
    }

    ReloadPropertyInformation();
    if (!new_map->is_dictionary_map()) return;
  }

  Handle<AccessorPair> pair;
  if (state() == ACCESSOR && GetAccessors()->IsAccessorPair()) {
    pair = Handle<AccessorPair>::cast(GetAccessors());
    // If the component and attributes are identical, nothing has to be done.
    if (pair->get(component) == *accessor) {
      if (property_details().attributes() == attributes) return;
    } else {
      pair = AccessorPair::Copy(pair);
      pair->set(component, *accessor);
    }
  } else {
    pair = factory()->NewAccessorPair();
    pair->set(component, *accessor);
  }

  TransitionToAccessorPair(pair, attributes);

#if VERIFY_HEAP
  if (FLAG_verify_heap) {
    receiver->JSObjectVerify();
  }
#endif
}
