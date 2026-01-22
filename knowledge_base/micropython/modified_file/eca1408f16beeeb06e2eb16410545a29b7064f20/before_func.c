            mp_print_str(print, "false");
        }
    } else {
        if (self->value) {
            mp_print_str(print, "True");
        } else {
            mp_print_str(print, "False");
        }
    }
}

STATIC mp_obj_t bool_make_new(const mp_obj_type_t *type_in, size_t n_args, size_t n_kw, const mp_obj_t *args) {
