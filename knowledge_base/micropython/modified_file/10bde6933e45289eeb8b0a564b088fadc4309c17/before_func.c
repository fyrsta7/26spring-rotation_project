    mp_int_t us = mp_obj_get_int(arg);
    if (us > 0) {
        MP_THREAD_GIL_EXIT();
        mp_hal_delay_us(us);
        MP_THREAD_GIL_ENTER();
    }
    return mp_const_none;
}
MP_DEFINE_CONST_FUN_OBJ_1(mp_utime_sleep_us_obj, time_sleep_us);

STATIC mp_obj_t time_ticks_ms(void) {
    return MP_OBJ_NEW_SMALL_INT(mp_hal_ticks_ms() & (MICROPY_PY_UTIME_TICKS_PERIOD - 1));
