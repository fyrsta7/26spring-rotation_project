	IHashingBuffer<Buffer>(size_t block_size_ = DBMS_DEFAULT_HASHING_BLOCK_SIZE)
		: BufferWithOwnMemory<Buffer>(block_size_), block_pos(0), block_size(block_size_), state(0, 0)
	{
	}
