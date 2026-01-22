        if (direction > 0) {
            str_index = 0;
            str_index_end = hlen - nlen;
        } else {
            str_index = hlen - nlen;
            str_index_end = 0;
        }
        for (;;) {
            if (memcmp(&haystack[str_index], needle, nlen) == 0) {
                //found
                return haystack + str_index;
            }
            if (str_index == str_index_end) {
                //not found
                break;
            }
            str_index += direction;
        }
    }
    return NULL;
}

// Note: this function is used to check if an object is a str or bytes, which
// works because both those types use it as their binary_op method.  Revisit
// MP_OBJ_IS_STR_OR_BYTES if this fact changes.
mp_obj_t mp_obj_str_binary_op(mp_uint_t op, mp_obj_t lhs_in, mp_obj_t rhs_in) {
    // check for modulo
    if (op == MP_BINARY_OP_MODULO) {
        mp_obj_t *args;
        mp_uint_t n_args;
        mp_obj_t dict = MP_OBJ_NULL;
        if (MP_OBJ_IS_TYPE(rhs_in, &mp_type_tuple)) {
            // TODO: Support tuple subclasses?
            mp_obj_tuple_get(rhs_in, &n_args, &args);
        } else if (MP_OBJ_IS_TYPE(rhs_in, &mp_type_dict)) {
            args = NULL;
            n_args = 0;
            dict = rhs_in;
        } else {
            args = &rhs_in;
            n_args = 1;
        }
        return str_modulo_format(lhs_in, n_args, args, dict);
    }

    // from now on we need lhs type and data, so extract them
    mp_obj_type_t *lhs_type = mp_obj_get_type(lhs_in);
    GET_STR_DATA_LEN(lhs_in, lhs_data, lhs_len);

    // check for multiply
    if (op == MP_BINARY_OP_MULTIPLY) {
        mp_int_t n;
        if (!mp_obj_get_int_maybe(rhs_in, &n)) {
            return MP_OBJ_NULL; // op not supported
        }
        if (n <= 0) {
            if (lhs_type == &mp_type_str) {
                return MP_OBJ_NEW_QSTR(MP_QSTR_); // empty str
            } else {
                return mp_const_empty_bytes;
            }
        }
        vstr_t vstr;
        vstr_init_len(&vstr, lhs_len * n);
        mp_seq_multiply(lhs_data, sizeof(*lhs_data), lhs_len, n, vstr.buf);
        return mp_obj_new_str_from_vstr(lhs_type, &vstr);
    }

    // From now on all operations allow:
    //    - str with str
    //    - bytes with bytes
    //    - bytes with bytearray
    //    - bytes with array.array
    // To do this efficiently we use the buffer protocol to extract the raw
    // data for the rhs, but only if the lhs is a bytes object.
    //
    // NOTE: CPython does not allow comparison between bytes ard array.array
    // (even if the array is of type 'b'), even though it allows addition of
    // such types.  We are not compatible with this (we do allow comparison
    // of bytes with anything that has the buffer protocol).  It would be
    // easy to "fix" this with a bit of extra logic below, but it costs code
    // size and execution time so we don't.

    const byte *rhs_data;
    mp_uint_t rhs_len;
    if (lhs_type == mp_obj_get_type(rhs_in)) {
        GET_STR_DATA_LEN(rhs_in, rhs_data_, rhs_len_);
        rhs_data = rhs_data_;
        rhs_len = rhs_len_;
    } else if (lhs_type == &mp_type_bytes) {
        mp_buffer_info_t bufinfo;
        if (!mp_get_buffer(rhs_in, &bufinfo, MP_BUFFER_READ)) {
            return MP_OBJ_NULL; // op not supported
        }
        rhs_data = bufinfo.buf;
        rhs_len = bufinfo.len;
    } else {
        // incompatible types
        return MP_OBJ_NULL; // op not supported
    }

    switch (op) {
        case MP_BINARY_OP_ADD:
        case MP_BINARY_OP_INPLACE_ADD: {
            if (lhs_len == 0) {
                return rhs_in;
            }
            if (rhs_len == 0) {
