void MergeTreeReader::Stream::loadMarks(MarkCache * cache, bool save_in_cache)
{
	std::string path = path_prefix + ".mrk";

	UInt128 key;
	if (cache)
	{
		key = cache->hash(path);
		marks = cache->get(key);
		if (marks)
			return;
	}

	size_t file_size = Poco::File(path).getSize();

	if (file_size % sizeof(MarkInCompressedFile) != 0)
		throw Exception("Size of " + path + " file is not divisable by size of MarkInCompressedFile structure.", ErrorCodes::CORRUPTED_DATA);

	size_t num_marks = file_size / sizeof(MarkInCompressedFile);

	marks = std::make_shared<MarksInCompressedFile>(num_marks);

	/// Read directly to marks.
	ReadBufferFromFile buffer(path, file_size, -1, reinterpret_cast<char *>(marks->data()));

	if (buffer.eof() || buffer.buffer().size() != file_size)
		throw Exception("Cannot read all marks from file " + path, ErrorCodes::CANNOT_READ_ALL_DATA);

	if (cache && save_in_cache)
		cache->set(key, marks);
}
