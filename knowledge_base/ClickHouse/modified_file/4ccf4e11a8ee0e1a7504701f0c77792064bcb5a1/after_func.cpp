void DefaultCoordinator::updateReadingState(InitialAllRangesAnnouncement announcement)
{
    PartRefs parts_diff;

    /// To get rid of duplicates
    for (auto && part_ranges: announcement.description)
    {
        Part part{.description = std::move(part_ranges), .replicas = {announcement.replica_num}};
        auto the_same_it = all_parts_to_read.find(part);
        /// We have the same part - add the info about presence on current replica to it
        if (the_same_it != all_parts_to_read.end())
        {
            the_same_it->replicas.insert(announcement.replica_num);
            continue;
        }

        auto covering_or_the_same_it = std::find_if(all_parts_to_read.begin(), all_parts_to_read.end(),
            [&part] (const Part & other) { return !other.description.info.isDisjoint(part.description.info); });

        /// It is covering part or we have covering - skip it
        if (covering_or_the_same_it != all_parts_to_read.end())
            continue;

        auto [insert_it, _] = all_parts_to_read.emplace(part);
        parts_diff.push_back(insert_it);
    }

    /// Split all parts by consistent hash
    while (!parts_diff.empty())
    {
        auto current_part_it = parts_diff.front();
        parts_diff.pop_front();
        auto consistent_hash = computeConsistentHash(current_part_it->description.info);

        /// Check whether the new part can easy go to replica queue
        if (current_part_it->replicas.contains(consistent_hash))
        {
            reading_state[consistent_hash].emplace_back(current_part_it);
            continue;
        }

        /// Add to delayed parts
        delayed_parts.emplace_back(current_part_it);
    }
}
