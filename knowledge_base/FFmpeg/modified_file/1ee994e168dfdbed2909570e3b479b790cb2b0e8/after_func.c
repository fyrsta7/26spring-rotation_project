    pc->state = 0;
    m->bytes_read = 0;
    m->ft = 0;
    m->skipped_codestream = 0;
    m->fheader_read = 0;
    m->skip_bytes = 0;
    m->read_tp = 0;
    m->in_codestream = 0;
}

/* Returns 1 if marker has any data which can be skipped
*/
static uint8_t info_marker(uint16_t marker)
{
    if (marker == 0xFF92 || marker == 0xFF4F ||
        marker == 0xFF90 || marker == 0xFF93 ||
        marker == 0xFFD9)
        return 0;
    else
        if (marker > 0xFF00) return 1;
    return 0;
}

/**
 * Find the end of the current frame in the bitstream.
 * @return the position of the first byte of the next frame, or -1
 */
static int find_frame_end(JPEG2000ParserContext *m, const uint8_t *buf, int buf_size)
{
    ParseContext *pc= &m->pc;
    int i;
    uint32_t state, next_state;
    uint64_t state64;
    state= pc->state;
    state64 = pc->state64;
    if (buf_size == 0) {
        return 0;
    }

    for (i = 0; i < buf_size; i++) {
        state = state << 8 | buf[i];
        state64 = state64 << 8 | buf[i];
        m->bytes_read++;
        if (m->skip_bytes) {
            // handle long skips
            if (m->skip_bytes > 8) {
                // need -9 else buf_size - i == 8 ==> i == buf_size after this,
                // and thus i == buf_size + 1 after the loop
                int skip = FFMIN(FFMIN((int64_t)m->skip_bytes - 8, buf_size - i - 9), INT_MAX);
                if (skip > 0) {
                    m->skip_bytes -= skip;
                    i += skip;
                    m->bytes_read += skip;
                }
            }
            m->skip_bytes--;
            continue;
        }
        if (m->read_tp) { // Find out how many bytes inside Tile part codestream to skip.
            if (m->read_tp == 1) {
                m->skip_bytes = (state64 & 0xFFFFFFFF) - 9 > 0?
                                (state64 & 0xFFFFFFFF) - 9 : 0;
            }
            m->read_tp--;
            continue;
        }
        if (m->fheader_read) {
            if (m->fheader_read == 1) {
                if (state64 == 0x6A5020200D0A870A) { // JP2 signature box value.
                    if (pc->frame_start_found) {
                        pc->frame_start_found = 0;
                        reset_context(m);
                        return i - 11;
                    } else {
                        pc->frame_start_found = 1;
                        m->ft = jp2_file;
                    }
                }
            }
            m->fheader_read--;
        }
        if (state == 0x0000000C && m->bytes_read >= 3) { // Indicates start of JP2 file. Check signature next.
            m->fheader_read = 8;
        } else if ((state & 0xFFFF) == 0xFF4F) {
            m->in_codestream = 1;
            if (!pc->frame_start_found) {
                pc->frame_start_found = 1;
                m->ft = j2k_cstream;
            } else if (pc->frame_start_found && m->ft == jp2_file && m->skipped_codestream) {
                reset_context(m);
                return i - 1;
            }
        } else if ((state & 0xFFFF) == 0xFFD9) {
            if (pc->frame_start_found && m->ft == jp2_file) {
