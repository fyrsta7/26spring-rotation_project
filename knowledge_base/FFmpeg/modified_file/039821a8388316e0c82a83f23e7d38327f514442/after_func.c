    // TODO : sbr_extension implementation
    ff_log_missing_feature(ac->avccontext, "SBR", 0);
    skip_bits_long(gb, 8*cnt - 4); // -4 due to reading extension type
    return cnt;
}

/**
 * Parse whether channels are to be excluded from Dynamic Range Compression; reference: table 4.53.
 *
 * @return  Returns number of bytes consumed.
