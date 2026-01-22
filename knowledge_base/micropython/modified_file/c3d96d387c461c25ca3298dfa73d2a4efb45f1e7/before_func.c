    mp_obj_exception_t *o = m_new_obj_var_maybe(mp_obj_exception_t, mp_obj_t, 0);
    if (o == NULL) {
        // Couldn't allocate heap memory; use local data instead.
        o = &MP_STATE_VM(mp_emergency_exception_obj);
        // We can't store any args.
        o->args = (mp_obj_tuple_t*)&mp_const_empty_tuple_obj;
    } else {
        o->args = MP_OBJ_TO_PTR(mp_obj_new_tuple(n_args, args));
    }
    o->base.type = type;
    o->traceback_data = NULL;
    return MP_OBJ_FROM_PTR(o);
