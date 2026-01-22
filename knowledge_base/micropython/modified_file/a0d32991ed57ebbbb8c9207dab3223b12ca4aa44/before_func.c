mp_obj_t mp_load_name(qstr qstr) {
    // logic: search locals, globals, builtins
    DEBUG_OP_printf("load name %s\n", qstr_str(qstr));
    mp_map_elem_t *elem = mp_map_lookup(map_locals, MP_OBJ_NEW_QSTR(qstr), MP_MAP_LOOKUP);
    if (elem != NULL) {
        return elem->value;
    } else {
        return mp_load_global(qstr);
    }
}
