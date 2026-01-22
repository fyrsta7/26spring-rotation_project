size_t MarksInCompressedFile::approximateMemoryUsage() const
{
    return sizeof(*this) + blocks.size() * sizeof(blocks[0]) + packed.size() * sizeof(packed[0]);
}
