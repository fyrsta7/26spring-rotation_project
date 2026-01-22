
constexpr void swap_keys(u32* keys, size_t i, size_t j)
{
    u32 temp = keys[i];
    keys[i] = keys[j];
    keys[j] = temp;
}

String AESCipherBlock::to_string() const
{
    StringBuilder builder;
    for (size_t i = 0; i < BlockSizeInBits / 8; ++i)
        builder.appendf("%02x", m_data[i]);
    return builder.build();
}

String AESCipherKey::to_string() const
{
    StringBuilder builder;
    for (size_t i = 0; i < (rounds() + 1) * 4; ++i)
        builder.appendf("%02x", m_rd_keys[i]);
    return builder.build();
}

void AESCipherKey::expand_encrypt_key(const ByteBuffer& user_key, size_t bits)
{
    u32* round_key;
    u32 temp;
    size_t i { 0 };

    ASSERT(!user_key.is_null());
    ASSERT(is_valid_key_size(bits));
    ASSERT(user_key.size() >= bits / 8);

    round_key = round_keys();

    if (bits == 128) {
        m_rounds = 10;
    } else if (bits == 192) {
        m_rounds = 12;
    } else {
        m_rounds = 14;
    }

    round_key[0] = get_key(user_key.data());
    round_key[1] = get_key(user_key.data() + 4);
    round_key[2] = get_key(user_key.data() + 8);
    round_key[3] = get_key(user_key.data() + 12);
    if (bits == 128) {
        for (;;) {
            temp = round_key[3];
            // clang-format off
            round_key[4] = round_key[0] ^
                    (AESTables::Encode2[(temp >> 16) & 0xff] & 0xff000000) ^
                    (AESTables::Encode3[(temp >>  8) & 0xff] & 0x00ff0000) ^
                    (AESTables::Encode0[(temp      ) & 0xff] & 0x0000ff00) ^
                    (AESTables::Encode1[(temp >> 24)       ] & 0x000000ff) ^ AESTables::RCON[i];
            // clang-format on
            round_key[5] = round_key[1] ^ round_key[4];
            round_key[6] = round_key[2] ^ round_key[5];
            round_key[7] = round_key[3] ^ round_key[6];
            ++i;
            if (i == 10)
                break;
            round_key += 4;
        }
        return;
    }

    round_key[4] = get_key(user_key.data() + 16);
    round_key[5] = get_key(user_key.data() + 20);
    if (bits == 192) {
        for (;;) {
            temp = round_key[5];
            // clang-format off
            round_key[6] = round_key[0] ^
                    (AESTables::Encode2[(temp >> 16) & 0xff] & 0xff000000) ^
                    (AESTables::Encode3[(temp >>  8) & 0xff] & 0x00ff0000) ^
                    (AESTables::Encode0[(temp      ) & 0xff] & 0x0000ff00) ^
                    (AESTables::Encode1[(temp >> 24)       ] & 0x000000ff) ^ AESTables::RCON[i];
            // clang-format on
            round_key[7] = round_key[1] ^ round_key[6];
            round_key[8] = round_key[2] ^ round_key[7];
            round_key[9] = round_key[3] ^ round_key[8];

            ++i;
            if (i == 8)
                break;

            round_key[10] = round_key[4] ^ round_key[9];
            round_key[11] = round_key[5] ^ round_key[10];

            round_key += 6;
        }
        return;
    }

    round_key[6] = get_key(user_key.data() + 24);
    round_key[7] = get_key(user_key.data() + 28);
    if (true) { // bits == 256
        for (;;) {
            temp = round_key[7];
            // clang-format off
            round_key[8] = round_key[0] ^
                    (AESTables::Encode2[(temp >> 16) & 0xff] & 0xff000000) ^
                    (AESTables::Encode3[(temp >>  8) & 0xff] & 0x00ff0000) ^
                    (AESTables::Encode0[(temp      ) & 0xff] & 0x0000ff00) ^
                    (AESTables::Encode1[(temp >> 24)       ] & 0x000000ff) ^ AESTables::RCON[i];
            // clang-format on
            round_key[9] = round_key[1] ^ round_key[8];
