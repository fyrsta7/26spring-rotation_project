    // TODO : sbr_extension implementation
    ff_log_missing_feature(ac->avccontext, "SBR", 0);
    skip_bits_long(gb, 8*cnt - 4); // -4 due to reading extension type
    return cnt;
}
