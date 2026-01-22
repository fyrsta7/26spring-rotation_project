mp_obj_t mp_load_name(qstr qstr) {
    // logic: search locals, globals, builtins
    DEBUG_OP_printf("load name %s\n", map_locals, qstr_str(qstr));
    // If we're at the outer scope (locals == globals), dispatch to load_global right away
    if (map_locals != map_globals) {
        mp_map_elem_t *elem = mp_map_lookup(map_locals, MP_OBJ_NEW_QSTR(qstr), MP_MAP_LOOKUP);
        if (elem != NULL) {
            return elem->value;
        }
    }
    return mp_load_global(qstr);
}
