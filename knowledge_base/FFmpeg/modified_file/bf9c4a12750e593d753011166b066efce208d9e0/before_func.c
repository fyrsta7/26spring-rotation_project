 * @param fc_cur (2.13) original fixed-codebook vector
 * @param gain_code (14.1) gain code
 * @param subframe_size length of the subframe
 */
static void g729d_get_new_exc(
        int16_t* out,
        const int16_t* in,
        const int16_t* fc_cur,
        int dstate,
