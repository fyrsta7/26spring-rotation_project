                hold[j] = 63;
            } else if (in[j] == '=') {
                if (j < 2 || i > 4) {
                    nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "invalid padding"));
                }
                hold[j] = 64;
            } else {
                nlr_raise(mp_obj_new_exception_msg(&mp_type_ValueError, "invalid character"));
            }
        }
        in += 4;

        *out++ = (hold[0]) << 2 | (hold[1]) >> 4;
        if (hold[2] != 64) {
            *out++ = (hold[1] & 0x0F) << 4 | hold[2] >> 2;
            if (hold[3] != 64) {
                *out++ = (hold[2] & 0x03) << 6 | hold[3];
            }
        }
    }
    return mp_obj_new_str_from_vstr(&mp_type_bytes, &vstr);
}
MP_DEFINE_CONST_FUN_OBJ_1(mod_binascii_a2b_base64_obj, mod_binascii_a2b_base64);

mp_obj_t mod_binascii_b2a_base64(mp_obj_t data) {
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(data, &bufinfo, MP_BUFFER_READ);

    vstr_t vstr;
    vstr_init_len(&vstr, ((bufinfo.len != 0) ? (((bufinfo.len - 1) / 3) + 1) * 4 : 0) + 1);

    // First pass, we convert input buffer to numeric base 64 values
    byte *in = bufinfo.buf, *out = (byte*)vstr.buf;
    mp_uint_t i;
    for (i = bufinfo.len; i >= 3; i -= 3) {
        *out++ = (in[0] & 0xFC) >> 2;
        *out++ = (in[0] & 0x03) << 4 | (in[1] & 0xF0) >> 4;
        *out++ = (in[1] & 0x0F) << 2 | (in[2] & 0xC0) >> 6;
        *out++ = in[2] & 0x3F;
        in += 3;
    }
    if (i != 0) {
        *out++ = (in[0] & 0xFC) >> 2;
        if (i == 2) {
            *out++ = (in[0] & 0x03) << 4 | (in[1] & 0xF0) >> 4;
            *out++ = (in[1] & 0x0F) << 2;
        }
        else {
            *out++ = (in[0] & 0x03) << 4;
            *out++ = 64;
        }
