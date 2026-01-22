        if (len < 0)
            return len;

        switch (type) {
        case AV1_OBU_SEQUENCE_HEADER:
            if (!obu_size)
                return AVERROR_INVALIDDATA;

            return parse_sequence_header(seq, buf + start_pos, obu_size);
        default:
            break;
        }
        size -= len;
        buf  += len;
    }

    return AVERROR_INVALIDDATA;
}

int ff_isom_write_av1c(AVIOContext *pb, const uint8_t *buf, int size)
{
    AVIOContext *seq_pb = NULL, *meta_pb = NULL;
    AV1SequenceParameters seq_params;
    PutBitContext pbc;
    uint8_t header[4];
    uint8_t *seq = NULL, *meta = NULL;
    int64_t obu_size;
    int start_pos, type, temporal_id, spatial_id;
    int ret, nb_seq = 0, seq_size, meta_size;

    if (size <= 0)
        return AVERROR_INVALIDDATA;

    ret = avio_open_dyn_buf(&seq_pb);
    if (ret < 0)
        return ret;
    ret = avio_open_dyn_buf(&meta_pb);
    if (ret < 0)
        goto fail;

    while (size > 0) {
        int len = parse_obu_header(buf, size, &obu_size, &start_pos,
                                   &type, &temporal_id, &spatial_id);
        if (len < 0) {
            ret = len;
            goto fail;
        }

        switch (type) {
        case AV1_OBU_SEQUENCE_HEADER:
            nb_seq++;
            if (!obu_size || nb_seq > 1) {
                ret = AVERROR_INVALIDDATA;
                goto fail;
            }
            ret = parse_sequence_header(&seq_params, buf + start_pos, obu_size);
            if (ret < 0)
                goto fail;

            avio_write(seq_pb, buf, len);
            break;
        case AV1_OBU_METADATA:
            if (!obu_size) {
                ret = AVERROR_INVALIDDATA;
                goto fail;
            }
            avio_write(meta_pb, buf, len);
            break;
        default:
            break;
        }
        size -= len;
        buf  += len;
    }

    seq_size  = avio_close_dyn_buf(seq_pb, &seq);
    if (!seq_size) {
        ret = AVERROR_INVALIDDATA;
        goto fail;
    }

    init_put_bits(&pbc, header, sizeof(header));

    put_bits(&pbc, 1, 1); // marker
    put_bits(&pbc, 7, 1); // version
    put_bits(&pbc, 3, seq_params.profile);
    put_bits(&pbc, 5, seq_params.level);
    put_bits(&pbc, 1, seq_params.tier);
    put_bits(&pbc, 1, seq_params.bitdepth > 8);
    put_bits(&pbc, 1, seq_params.bitdepth == 12);
    put_bits(&pbc, 1, seq_params.monochrome);
    put_bits(&pbc, 1, seq_params.chroma_subsampling_x);
    put_bits(&pbc, 1, seq_params.chroma_subsampling_y);
    put_bits(&pbc, 2, seq_params.chroma_sample_position);
    put_bits(&pbc, 8, 0); // padding
