
    *emit_get_cur_to_write_code_info(emit, 1) = 0; // end of line number info
    emit_align_code_info_to_machine_word(emit); // align so that following bytecode is aligned

    if (emit->pass == MP_PASS_CODE_SIZE) {
        // calculate size of code in bytes
        emit->code_info_size = emit->code_info_offset;
        emit->bytecode_size = emit->bytecode_offset;
        emit->code_base = m_new0(byte, emit->code_info_size + emit->bytecode_size);

    } else if (emit->pass == MP_PASS_EMIT) {
        qstr *arg_names = m_new(qstr, emit->scope->num_pos_args + emit->scope->num_kwonly_args);
        for (int i = 0; i < emit->scope->num_pos_args + emit->scope->num_kwonly_args; i++) {
