#endif


#ifdef MATRIX_HAS_GHOST
extern const uint16_t keymaps[][MATRIX_ROWS][MATRIX_COLS];
static matrix_row_t get_real_keys(uint8_t row, matrix_row_t rowdata){
    matrix_row_t out = 0;
    for (int col = 0; col < MATRIX_COLS; col++) {
        if (pgm_read_byte(&keymaps[0][row][col]) && ((rowdata & (1<<col)))){
            out |= 1<<col;
        }
    }
