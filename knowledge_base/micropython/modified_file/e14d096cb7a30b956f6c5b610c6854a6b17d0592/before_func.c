    }
    return MP_BOOL(memcmp(str, prefix, prefix_len) == 0);
}

enum { LSTRIP, RSTRIP, STRIP };

STATIC mp_obj_t str_uni_strip(int type, uint n_args, const mp_obj_t *args) {
    assert(1 <= n_args && n_args <= 2);
    assert(MP_OBJ_IS_STR(args[0]));

    const byte *chars_to_del;
    uint chars_to_del_len;
    static const byte whitespace[] = " \t\n\r\v\f";

    if (n_args == 1) {
        chars_to_del = whitespace;
        chars_to_del_len = sizeof(whitespace);
    } else {
        assert(MP_OBJ_IS_STR(args[1]));
        GET_STR_DATA_LEN(args[1], s, l);
        chars_to_del = s;
        chars_to_del_len = l;
    }

    GET_STR_DATA_LEN(args[0], orig_str, orig_str_len);

    machine_uint_t first_good_char_pos = 0;
    bool first_good_char_pos_set = false;
    machine_uint_t last_good_char_pos = 0;
    // TODO: For RSPLIT, scan from end
    for (machine_uint_t i = 0; i < orig_str_len; i++) {
        if (find_subbytes(chars_to_del, chars_to_del_len, &orig_str[i], 1, 1) == NULL) {
            if (!first_good_char_pos_set) {
                first_good_char_pos = i;
                if (type == LSTRIP) {
                    last_good_char_pos = orig_str_len - 1;
                    break;
                }
                first_good_char_pos_set = true;
            }
            last_good_char_pos = i;
        }
    }

    if (first_good_char_pos == 0 && last_good_char_pos == 0) {
        // string is all whitespace, return ''
        return MP_OBJ_NEW_QSTR(MP_QSTR_);
    }

    assert(last_good_char_pos >= first_good_char_pos);
    if (type == RSTRIP) {
