MergeTreeSequentialSource::MergeTreeSequentialSource(
    MergeTreeSequentialSourceType type,
    const MergeTreeData & storage_,
    const StorageSnapshotPtr & storage_snapshot_,
    MergeTreeData::DataPartPtr data_part_,
    AlterConversionsPtr alter_conversions_,
    Names columns_to_read_,
    std::optional<MarkRanges> mark_ranges_,
    bool apply_deleted_mask,
    bool read_with_direct_io_,
    bool prefetch)
    : ISource(storage_snapshot_->getSampleBlockForColumns(columns_to_read_))
    , storage(storage_)
    , storage_snapshot(storage_snapshot_)
    , data_part(std::move(data_part_))
    , alter_conversions(std::move(alter_conversions_))
    , columns_to_read(std::move(columns_to_read_))
    , read_with_direct_io(read_with_direct_io_)
    , mark_ranges(std::move(mark_ranges_))
    , mark_cache(storage.getContext()->getMarkCache())
{
    /// Print column name but don't pollute logs in case of many columns.
    if (columns_to_read.size() == 1)
        LOG_DEBUG(log, "Reading {} marks from part {}, total {} rows starting from the beginning of the part, column {}",
            data_part->getMarksCount(), data_part->name, data_part->rows_count, columns_to_read.front());
    else
        LOG_DEBUG(log, "Reading {} marks from part {}, total {} rows starting from the beginning of the part",
            data_part->getMarksCount(), data_part->name, data_part->rows_count);

    /// Note, that we don't check setting collaborate_with_coordinator presence, because this source
    /// is only used in background merges.
    addTotalRowsApprox(data_part->rows_count);

    /// Add columns because we don't want to read empty blocks
    injectRequiredColumns(
        LoadedMergeTreeDataPartInfoForReader(data_part, alter_conversions),
        storage_snapshot,
        storage.supportsSubcolumns(),
        columns_to_read);

    auto options = GetColumnsOptions(GetColumnsOptions::AllPhysical)
        .withExtendedObjects()
        .withVirtuals()
        .withSubcolumns(storage.supportsSubcolumns());

    auto columns_for_reader = storage_snapshot->getColumnsByNames(options, columns_to_read);

    const auto & context = storage.getContext();
    ReadSettings read_settings = context->getReadSettings();
    read_settings.read_from_filesystem_cache_if_exists_otherwise_bypass_cache = !(*storage.getSettings())[MergeTreeSetting::force_read_through_cache_for_merges];

    /// It does not make sense to use pthread_threadpool for background merges/mutations
    /// And also to preserve backward compatibility
    read_settings.local_fs_method = LocalFSReadMethod::pread;
    if (read_with_direct_io)
        read_settings.direct_io_threshold = 1;

    /// Configure throttling
    switch (type)
    {
        case Mutation:
            read_settings.local_throttler = context->getMutationsThrottler();
            break;
        case Merge:
            read_settings.local_throttler = context->getMergesThrottler();
            break;
    }
    read_settings.remote_throttler = read_settings.local_throttler;

    MergeTreeReaderSettings reader_settings =
    {
        .read_settings = read_settings,
        .save_marks_in_cache = false,
        .apply_deleted_mask = apply_deleted_mask,
        .can_read_part_without_marks = true,
    };

    if (!mark_ranges)
        mark_ranges.emplace(MarkRanges{MarkRange(0, data_part->getMarksCount())});

    reader = data_part->getReader(
        columns_for_reader,
        storage_snapshot,
        *mark_ranges,
        /*virtual_fields=*/ {},
        /*uncompressed_cache=*/ {},
        mark_cache.get(),
        alter_conversions,
        reader_settings,
        /*avg_value_size_hints=*/ {},
        /*profile_callback=*/ {});

    if (prefetch)
        reader->prefetchBeginOfRange(Priority{});
}
