size_t MarksInCompressedFile::approximateMemoryUsage() const
{
    return sizeof(*this) + blocks.allocated_bytes() + packed.allocated_bytes();
}
