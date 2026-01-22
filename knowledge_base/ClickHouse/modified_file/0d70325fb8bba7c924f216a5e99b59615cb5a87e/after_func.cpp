void CompressionCodecLZ4::doDecompressData(const char * source, UInt32, char * dest, UInt32 uncompressed_size) const
{
    if (LZ4_decompress_fast(source, dest, uncompressed_size) < 0)
        throw Exception("Cannot LZ4_decompress_safe", ErrorCodes::CANNOT_DECOMPRESS);
    // LZ4::decompress(source, dest, source_size, uncompressed_size, lz4_stat);
}
