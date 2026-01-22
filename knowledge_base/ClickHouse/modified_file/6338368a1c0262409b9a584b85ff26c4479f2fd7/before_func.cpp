            return remote_fs_segment_reader;
        }
        case ReadType::REMOTE_FS_READ_BYPASS_CACHE:
        {
            /// Result buffer is owned only by current buffer -- not shareable like in the case above.

            if (remote_file_reader && remote_file_reader->getFileOffsetOfBufferEnd() == file_offset_of_buffer_end)
                return remote_file_reader;

            remote_file_reader = remote_file_reader_creator();
            return remote_file_reader;
        }
        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Cannot use remote filesystem reader with read type: {}", toString(read_type));
    }
}

SeekableReadBufferPtr CachedReadBufferFromRemoteFS::getReadBufferForFileSegment(FileSegmentPtr & file_segment)
{
    auto range = file_segment->range();

    size_t wait_download_max_tries = settings.filesystem_cache_max_wait_sec;
    size_t wait_download_tries = 0;

    auto download_state = file_segment->state();

    if (settings.read_from_filesystem_cache_if_exists_otherwise_bypass_cache)
    {
        if (download_state == FileSegment::State::DOWNLOADED)
        {
            read_type = ReadType::CACHED;
            return getCacheReadBuffer(range.left);
        }
        else
        {
            read_type = ReadType::REMOTE_FS_READ_BYPASS_CACHE;
            return getRemoteFSReadBuffer(file_segment, read_type);
        }
    }

    while (true)
    {
        switch (download_state)
        {
            case FileSegment::State::SKIP_CACHE:
            {
                read_type = ReadType::REMOTE_FS_READ_BYPASS_CACHE;
                return getRemoteFSReadBuffer(file_segment, read_type);
            }
            case FileSegment::State::EMPTY:
            {
                auto downloader_id = file_segment->getOrSetDownloader();
                if (downloader_id == file_segment->getCallerId())
                {
                    if (file_offset_of_buffer_end == file_segment->getDownloadOffset())
                    {
                        read_type = ReadType::REMOTE_FS_READ_AND_PUT_IN_CACHE;
                        return getRemoteFSReadBuffer(file_segment, read_type);
                    }
                    else
                    {
                        ///                      segment{k}
                        /// cache:           [______|___________
                        ///                         ^
                        ///                         download_offset
                        /// requested_range:            [__________]
                        ///                             ^
                        ///                             file_offset_of_buffer_end
                        assert(file_offset_of_buffer_end > file_segment->getDownloadOffset());
                        bytes_to_predownload = file_offset_of_buffer_end - file_segment->getDownloadOffset();

                        read_type = ReadType::REMOTE_FS_READ_AND_PUT_IN_CACHE;
                        return getRemoteFSReadBuffer(file_segment, read_type);
                    }
                }
                else
                {
                    download_state = file_segment->state();
                    continue;
                }
            }
            case FileSegment::State::DOWNLOADING:
            {
                size_t download_offset = file_segment->getDownloadOffset();
                bool can_start_from_cache = download_offset > file_offset_of_buffer_end;

                /// If file segment is being downloaded but we can already read from already downloaded part, do that.
                if (can_start_from_cache)
                {
                    ///                      segment{k} state: DOWNLOADING
                    /// cache:           [______|___________
                    ///                         ^
                    ///                         download_offset (in progress)
                    /// requested_range:    [__________]
                    ///                     ^
                    ///                     file_offset_of_buffer_end

                    read_type = ReadType::CACHED;
                    return getCacheReadBuffer(range.left);
                }

                if (wait_download_tries++ < wait_download_max_tries)
                {
                    download_state = file_segment->wait();
                }
                else
                {
                    download_state = FileSegment::State::SKIP_CACHE;
                }

                continue;
            }
            case FileSegment::State::DOWNLOADED:
            {
                read_type = ReadType::CACHED;
                return getCacheReadBuffer(range.left);
            }
            case FileSegment::State::PARTIALLY_DOWNLOADED:
            {
                auto downloader_id = file_segment->getOrSetDownloader();
                if (downloader_id == file_segment->getCallerId())
                {
                    size_t download_offset = file_segment->getDownloadOffset();
                    bool can_start_from_cache = download_offset > file_offset_of_buffer_end;

                    LOG_TEST(log, "Current download offset: {}, file offset of buffer end: {}", download_offset, file_offset_of_buffer_end);

                    if (can_start_from_cache)
                    {
                        ///                      segment{k}
                        /// cache:           [______|___________
                        ///                         ^
                        ///                         download_offset
                        /// requested_range:    [__________]
                        ///                     ^
                        ///                     file_offset_of_buffer_end

                        read_type = ReadType::CACHED;
                        file_segment->resetDownloader();
                        return getCacheReadBuffer(range.left);
                    }

                    if (download_offset < file_offset_of_buffer_end)
                    {
                        ///                   segment{1}
                        /// cache:         [_____|___________
                        ///                      ^
                        ///                      download_offset
                        /// requested_range:          [__________]
                        ///                           ^
                        ///                           file_offset_of_buffer_end

                        assert(file_offset_of_buffer_end > file_segment->getDownloadOffset());
                        bytes_to_predownload = file_offset_of_buffer_end - file_segment->getDownloadOffset();
                    }

                    download_offset = file_segment->getDownloadOffset();
                    can_start_from_cache = download_offset > file_offset_of_buffer_end;
                    assert(!can_start_from_cache);

                    read_type = ReadType::REMOTE_FS_READ_AND_PUT_IN_CACHE;
                    return getRemoteFSReadBuffer(file_segment, read_type);
                }

                download_state = file_segment->state();
                continue;
            }
            case FileSegment::State::PARTIALLY_DOWNLOADED_NO_CONTINUATION:
            {
