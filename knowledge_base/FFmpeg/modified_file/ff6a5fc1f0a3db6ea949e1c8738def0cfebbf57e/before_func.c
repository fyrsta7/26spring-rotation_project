    { MATROSKA_ID_INFO,           EBML_NONE },
    { MATROSKA_ID_CUES,           EBML_NONE },
    { MATROSKA_ID_TAGS,           EBML_NONE },
    { MATROSKA_ID_SEEKHEAD,       EBML_NONE },
    { 0 }
};

static const char *matroska_doctypes[] = { "matroska", "webm" };

/*
 * Return: Whether we reached the end of a level in the hierarchy or not.
 */
static int ebml_level_end(MatroskaDemuxContext *matroska)
{
    ByteIOContext *pb = matroska->ctx->pb;
    int64_t pos = url_ftell(pb);

    if (matroska->num_levels > 0) {
        MatroskaLevel *level = &matroska->levels[matroska->num_levels - 1];
        if (pos - level->start >= level->length || matroska->current_id) {
            matroska->num_levels--;
            return 1;
        }
    }
    return 0;
}

/*
 * Read: an "EBML number", which is defined as a variable-length
 * array of bytes. The first byte indicates the length by giving a
 * number of 0-bits followed by a one. The position of the first
 * "one" bit inside the first byte indicates the length of this
 * number.
 * Returns: number of bytes read, < 0 on error
 */
static int ebml_read_num(MatroskaDemuxContext *matroska, ByteIOContext *pb,
                         int max_size, uint64_t *number)
{
    int len_mask = 0x80, read = 1, n = 1;
    int64_t total = 0;
