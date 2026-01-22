            mp_print_str(print, "false");
        }
    } else {
        if (self->value) {
            mp_print_str(print, "True");
        } else {
            mp_print_str(print, "False");
