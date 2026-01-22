    }

    return code;
}

ErrorOr<u32> CanonicalCode::read_symbol(LittleEndianInputBitStream& stream) const
{
    u32 code_bits = 1;

    for (;;) {
        code_bits = code_bits << 1 | TRY(stream.read_bit());
        if (code_bits >= (1 << 16))
            return Error::from_string_literal("Symbol exceeds maximum symbol number");

        // FIXME: This is very inefficient and could greatly be improved by implementing this
        //        algorithm: https://www.hanshq.net/zip.html#huffdec
