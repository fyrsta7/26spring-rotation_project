bool IC::UpdatePolymorphicIC(Handle<Name> name,
                             const MaybeObjectHandle& handler) {
  DCHECK(IsHandler(*handler));
  if (is_keyed() && state() != RECOMPUTE_HANDLER) {
    if (nexus()->GetName() != *name) return false;
  }
  Handle<Map> map = lookup_start_object_map();

  std::vector<MapAndHandler> maps_and_handlers;
  maps_and_handlers.reserve(v8_flags.max_valid_polymorphic_map_count);
  int deprecated_maps = 0;
  int handler_to_overwrite = -1;

  {
    DisallowGarbageCollection no_gc;
    int i = 0;
    for (FeedbackIterator it(nexus()); !it.done(); it.Advance()) {
      if (it.handler()->IsCleared()) continue;
      MaybeObjectHandle existing_handler = handle(it.handler(), isolate());
      Handle<Map> existing_map = handle(it.map(), isolate());

      maps_and_handlers.push_back(
          MapAndHandler(existing_map, existing_handler));

      if (existing_map->is_deprecated()) {
        // Filter out deprecated maps to ensure their instances get migrated.
        deprecated_maps++;
      } else if (map.is_identical_to(existing_map)) {
        // If both map and handler stayed the same (and the name is also the
        // same as checked above, for keyed accesses), we're not progressing
        // in the lattice and need to go MEGAMORPHIC instead. There's one
        // exception to this rule, which is when we're in RECOMPUTE_HANDLER
        // state, there we allow to migrate to a new handler.
        if (handler.is_identical_to(existing_handler) &&
            state() != RECOMPUTE_HANDLER) {
          return false;
        }

        // If the receiver type is already in the polymorphic IC, this indicates
        // there was a prototoype chain failure. In that case, just overwrite
        // the handler.
        handler_to_overwrite = i;
      } else if (handler_to_overwrite == -1 &&
                 IsTransitionOfMonomorphicTarget(*existing_map, *map)) {
        handler_to_overwrite = i;
      }

      i++;
    }
    DCHECK_LE(i, maps_and_handlers.size());
  }

  int number_of_maps = static_cast<int>(maps_and_handlers.size());
  int number_of_valid_maps =
      number_of_maps - deprecated_maps - (handler_to_overwrite != -1);

  if (number_of_valid_maps >= v8_flags.max_valid_polymorphic_map_count)
    return false;
  if (number_of_maps == 0 && state() != MONOMORPHIC && state() != POLYMORPHIC) {
    return false;
  }

  number_of_valid_maps++;
  if (number_of_valid_maps == 1) {
    ConfigureVectorState(name, lookup_start_object_map(), handler);
  } else {
    if (is_keyed() && nexus()->GetName() != *name) return false;
    if (handler_to_overwrite >= 0) {
      maps_and_handlers[handler_to_overwrite].second = handler;
      if (!map.is_identical_to(
              maps_and_handlers.at(handler_to_overwrite).first)) {
        maps_and_handlers[handler_to_overwrite].first = map;
      }
    } else {
      maps_and_handlers.push_back(MapAndHandler(map, handler));
    }

    ConfigureVectorState(name, maps_and_handlers);
  }

  return true;
}
