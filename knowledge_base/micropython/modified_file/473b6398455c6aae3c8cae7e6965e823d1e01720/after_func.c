    const mp_stream_p_t *sock_stream = mp_get_stream_raise(websock, MP_STREAM_OP_WRITE | MP_STREAM_OP_IOCTL);
    int err;
    int old_opts = sock_stream->ioctl(websock, MP_STREAM_SET_DATA_OPTS, FRAME_BIN, &err);
    sock_stream->write(websock, buf, len, &err);
    sock_stream->ioctl(websock, MP_STREAM_SET_DATA_OPTS, old_opts, &err);
}

STATIC void write_webrepl_resp(mp_obj_t websock, uint16_t code) {
    char buf[4] = {'W', 'B', code & 0xff, code >> 8};
    write_webrepl(websock, buf, sizeof(buf));
}

STATIC mp_obj_t webrepl_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 1, 2, false);
    mp_get_stream_raise(args[0], MP_STREAM_OP_READ | MP_STREAM_OP_WRITE | MP_STREAM_OP_IOCTL);
    DEBUG_printf("sizeof(struct webrepl_file) = %lu\n", sizeof(struct webrepl_file));
    mp_obj_webrepl_t *o = m_new_obj(mp_obj_webrepl_t);
    o->base.type = type;
    o->sock = args[0];
    o->hdr_to_recv = sizeof(struct webrepl_file);
    o->data_to_recv = 0;
    return o;
}

STATIC void handle_op(mp_obj_webrepl_t *self) {
    mp_obj_t open_args[2] = {
        mp_obj_new_str(self->hdr.fname, strlen(self->hdr.fname), false),
        MP_OBJ_NEW_QSTR(MP_QSTR_rb)
    };

    if (self->hdr.type == PUT_FILE) {
        open_args[1] = MP_OBJ_NEW_QSTR(MP_QSTR_wb);
    }

    self->cur_file = mp_builtin_open(2, open_args, (mp_map_t*)&mp_const_empty_map);
    const mp_stream_p_t *file_stream =
        mp_get_stream_raise(self->cur_file, MP_STREAM_OP_READ | MP_STREAM_OP_WRITE | MP_STREAM_OP_IOCTL);

    #if 0
    struct mp_stream_seek_t seek = { .offset = self->hdr.offset, .whence = 0 };
    int err;
    mp_uint_t res = file_stream->ioctl(self->cur_file, MP_STREAM_SEEK, (uintptr_t)&seek, &err);
    assert(res != MP_STREAM_ERROR);
    #endif

    write_webrepl_resp(self->sock, 0);
