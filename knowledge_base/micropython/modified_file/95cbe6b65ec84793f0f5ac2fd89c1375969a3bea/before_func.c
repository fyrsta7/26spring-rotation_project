    mp_obj_class_lookup(&lookup, lhs->base.type);

    mp_obj_t res;
    if (dest[0] == MP_OBJ_SENTINEL) {
        res = mp_binary_op(op, lhs->subobj[0], rhs_in);
    } else if (dest[0] != MP_OBJ_NULL) {
        dest[2] = rhs_in;
        res = mp_call_method_n_kw(1, 0, dest);
    } else {
        // If this was an inplace method, fallback to normal method
        // https://docs.python.org/3/reference/datamodel.html#object.__iadd__ :
        // "If a specific method is not defined, the augmented assignment
        // falls back to the normal methods."
        if (op >= MP_BINARY_OP_INPLACE_OR && op <= MP_BINARY_OP_INPLACE_POWER) {
            op -= MP_BINARY_OP_INPLACE_OR - MP_BINARY_OP_OR;
            goto retry;
        }
        return MP_OBJ_NULL; // op not supported
    }

    #if MICROPY_PY_BUILTINS_NOTIMPLEMENTED
    // NotImplemented means "try other fallbacks (like calling __rop__
    // instead of __op__) and if nothing works, raise TypeError". As
    // MicroPython doesn't implement any fallbacks, signal to raise
    // TypeError right away.
    if (res == mp_const_notimplemented) {
        return MP_OBJ_NULL; // op not supported
    }
    #endif

    return res;
}

STATIC void mp_obj_instance_load_attr(mp_obj_t self_in, qstr attr, mp_obj_t *dest) {
    // logic: look in instance members then class locals
    assert(mp_obj_is_instance_type(mp_obj_get_type(self_in)));
    mp_obj_instance_t *self = MP_OBJ_TO_PTR(self_in);

    mp_map_elem_t *elem = mp_map_lookup(&self->members, MP_OBJ_NEW_QSTR(attr), MP_MAP_LOOKUP);
    if (elem != NULL) {
        // object member, always treated as a value
        dest[0] = elem->value;
        return;
    }
    #if MICROPY_CPYTHON_COMPAT
    if (attr == MP_QSTR___dict__) {
        // Create a new dict with a copy of the instance's map items.
        // This creates, unlike CPython, a 'read-only' __dict__: modifying
        // it will not result in modifications to the actual instance members.
        mp_map_t *map = &self->members;
        mp_obj_t attr_dict = mp_obj_new_dict(map->used);
        for (size_t i = 0; i < map->alloc; ++i) {
            if (mp_map_slot_is_filled(map, i)) {
                mp_obj_dict_store(attr_dict, map->table[i].key, map->table[i].value);
            }
        }
        dest[0] = attr_dict;
        return;
    }
    #endif
    struct class_lookup_data lookup = {
        .obj = self,
        .attr = attr,
        .meth_offset = 0,
        .dest = dest,
        .is_type = false,
    };
    mp_obj_class_lookup(&lookup, self->base.type);
    mp_obj_t member = dest[0];
    if (member != MP_OBJ_NULL) {
        if (!(self->base.type->flags & MP_TYPE_FLAG_HAS_SPECIAL_ACCESSORS)) {
            // Class doesn't have any special accessors to check so return straightaway
            return;
        }

        #if MICROPY_PY_BUILTINS_PROPERTY
        if (mp_obj_is_type(member, &mp_type_property)) {
            // object member is a property; delegate the load to the property
            // Note: This is an optimisation for code size and execution time.
            // The proper way to do it is have the functionality just below
            // in a __get__ method of the property object, and then it would
            // be called by the descriptor code down below.  But that way
            // requires overhead for the nested mp_call's and overhead for
            // the code.
            const mp_obj_t *proxy = mp_obj_property_get(member);
            if (proxy[0] == mp_const_none) {
                mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("unreadable attribute"));
            } else {
                dest[0] = mp_call_function_n_kw(proxy[0], 1, 0, &self_in);
            }
            return;
        }
        #endif

        #if MICROPY_PY_DESCRIPTORS
        // found a class attribute; if it has a __get__ method then call it with the
        // class instance and class as arguments and return the result
        // Note that this is functionally correct but very slow: each load_attr
