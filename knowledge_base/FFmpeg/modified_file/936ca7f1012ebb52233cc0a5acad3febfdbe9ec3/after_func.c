        return AVERROR_INVALIDDATA;
    }
    bytestream2_skip(&ctx->gb, 8); // skip pad

    hdr->width  = bytestream2_get_le32u(&ctx->gb);
    hdr->height = bytestream2_get_le32u(&ctx->gb);

