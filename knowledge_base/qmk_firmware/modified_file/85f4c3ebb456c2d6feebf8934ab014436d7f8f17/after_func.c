 * FIXME: needs doc
 */
void keyboard_init(void) {
    timer_init();
    matrix_init();
#ifdef QWIIC_ENABLE
    qwiic_init();
#endif
#ifdef OLED_DRIVER_ENABLE
    oled_init(OLED_ROTATION_0);
#endif
#ifdef PS2_MOUSE_ENABLE
    ps2_mouse_init();
#endif
#ifdef SERIAL_MOUSE_ENABLE
    serial_mouse_init();
#endif
#ifdef ADB_MOUSE_ENABLE
    adb_mouse_init();
#endif
#ifdef BOOTMAGIC_ENABLE
    bootmagic();
#else
    magic();
#endif
#ifdef BACKLIGHT_ENABLE
    backlight_init();
#endif
#ifdef RGBLIGHT_ENABLE
    rgblight_init();
#endif
#ifdef STENO_ENABLE
    steno_init();
#endif
#ifdef FAUXCLICKY_ENABLE
    fauxclicky_init();
#endif
#ifdef POINTING_DEVICE_ENABLE
    pointing_device_init();
#endif
#if defined(NKRO_ENABLE) && defined(FORCE_NKRO)
    keymap_config.nkro = 1;
    eeconfig_update_keymap(keymap_config.raw);
#endif
    keyboard_post_init_kb(); /* Always keep this last */
}

/** \brief Keyboard task: Do keyboard routine jobs
 *
 * Do routine keyboard jobs:
 *
 * * scan matrix
 * * handle mouse movements
 * * run visualizer code
 * * handle midi commands
 * * light LEDs
 *
 * This is repeatedly called as fast as possible.
 */
void keyboard_task(void) {
    static matrix_row_t matrix_prev[MATRIX_ROWS];
    static uint8_t      led_status    = 0;
    matrix_row_t        matrix_row    = 0;
    matrix_row_t        matrix_change = 0;
#ifdef QMK_KEYS_PER_SCAN
    uint8_t keys_processed = 0;
#endif

#if defined(OLED_DRIVER_ENABLE) && !defined(OLED_DISABLE_TIMEOUT)
    uint8_t ret = matrix_scan();
#else
    matrix_scan();
#endif

    if (is_keyboard_master()) {
        for (uint8_t r = 0; r < MATRIX_ROWS; r++) {
            matrix_row    = matrix_get_row(r);
            matrix_change = matrix_row ^ matrix_prev[r];
            if (matrix_change) {
#ifdef MATRIX_HAS_GHOST
                if (has_ghost_in_row(r, matrix_row)) {
                    continue;
                }
#endif
                if (debug_matrix) matrix_print();
                matrix_row_t col_mask = 1;
                for (uint8_t c = 0; c < MATRIX_COLS; c++, col_mask <<= 1) {
                    if (matrix_change & col_mask) {
                        action_exec((keyevent_t){
                            .key = (keypos_t){.row = r, .col = c}, .pressed = (matrix_row & col_mask), .time = (timer_read() | 1) /* time should not be 0 */
                        });
                        // record a processed key
                        matrix_prev[r] ^= col_mask;
#ifdef QMK_KEYS_PER_SCAN
                        // only jump out if we have processed "enough" keys.
                        if (++keys_processed >= QMK_KEYS_PER_SCAN)
#endif
                            // process a key per task call
                            goto MATRIX_LOOP_END;
                    }
                }
            }
        }
    }
    // call with pseudo tick event when no real key event.
#ifdef QMK_KEYS_PER_SCAN
    // we can get here with some keys processed now.
    if (!keys_processed)
#endif
        action_exec(TICK);

MATRIX_LOOP_END:

#ifdef DEBUG_MATRIX_SCAN_RATE
    matrix_scan_perf_task();
