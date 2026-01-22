void ReplicatedMergeTreeQueue::insertUnlocked(LogEntryPtr & entry, std::optional<time_t> & min_unprocessed_insert_time_changed, std::lock_guard<std::mutex> &)
{
    virtual_parts.add(entry->new_part_name);

    /// Put 'DROP PARTITION' entries at the beginning of the queue not to make superfluous fetches of parts that will be eventually deleted
    if (entry->type != LogEntry::DROP_RANGE)
        queue.push_back(entry);
    else
        queue.push_front(entry);

    if (entry->type == LogEntry::GET_PART)
    {
        inserts_by_time.insert(entry);

        if (entry->create_time && (!min_unprocessed_insert_time || entry->create_time < min_unprocessed_insert_time))
        {
            min_unprocessed_insert_time = entry->create_time;
            min_unprocessed_insert_time_changed = min_unprocessed_insert_time;
        }
    }
}
