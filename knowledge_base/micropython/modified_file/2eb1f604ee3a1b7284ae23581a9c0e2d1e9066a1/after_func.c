        case MP_BINARY_OP_MORE_EQUAL:
            if (lhs_type == rhs_type) {
                GET_STR_DATA_LEN(rhs_in, rhs_data, rhs_len);
                return MP_BOOL(mp_seq_cmp_bytes(op, lhs_data, lhs_len, rhs_data, rhs_len));
            }
            if (lhs_type == &mp_type_bytes) {
                mp_buffer_info_t bufinfo;
                if (!mp_get_buffer(rhs_in, &bufinfo, MP_BUFFER_READ)) {
                    goto uncomparable;
                }
                return MP_BOOL(mp_seq_cmp_bytes(op, lhs_data, lhs_len, bufinfo.buf, bufinfo.len));
            }
uncomparable:
            if (op == MP_BINARY_OP_EQUAL) {
                return mp_const_false;
            }
    }

    return MP_OBJ_NULL; // op not supported
}

#if !MICROPY_PY_BUILTINS_STR_UNICODE
// objstrunicode defines own version
const byte *str_index_to_ptr(const mp_obj_type_t *type, const byte *self_data, uint self_len,
                             mp_obj_t index, bool is_slice) {
    mp_uint_t index_val = mp_get_index(type, self_len, index, is_slice);
