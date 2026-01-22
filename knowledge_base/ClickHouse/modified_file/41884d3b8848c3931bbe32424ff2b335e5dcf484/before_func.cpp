uint64_t MetadataStorageFromPlainObjectStorage::getFileSize(const String & path) const
{
    RelativePathsWithSize children;
    object_storage->findAllFiles(getAbsolutePath(path), children);
    if (children.empty())
        return 0;
    if (children.size() != 1)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "findAllFiles() return multiple paths ({}) for {}", children.size(), path);
    return children.front().bytes_size;
}
