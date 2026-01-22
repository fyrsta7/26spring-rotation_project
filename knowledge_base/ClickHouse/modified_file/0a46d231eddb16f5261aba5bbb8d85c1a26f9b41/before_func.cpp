        /// (OPTIMIZE queries) can assign new merges.
        std::lock_guard<std::mutex> merge_selecting_lock(merge_selecting_mutex);

        auto zookeeper = getZooKeeper();

        ReplicatedMergeTreeMergePredicate merge_pred = queue.getMergePredicate(zookeeper);

        /// If many merges is already queued, then will queue only small enough merges.
        /// Otherwise merge queue could be filled with only large merges,
        /// and in the same time, many small parts could be created and won't be merged.
        size_t merges_and_mutations_queued = merge_pred.countMergesAndPartMutations();
        if (merges_and_mutations_queued >= data.settings.max_replicated_merges_in_queue)
        {
            LOG_TRACE(log, "Number of queued merges and part mutations (" << merges_and_mutations_queued
                << ") is greater than max_replicated_merges_in_queue ("
                << data.settings.max_replicated_merges_in_queue << "), so won't select new parts to merge or mutate.");
        }
        else
        {
            size_t max_source_parts_size = merger_mutator.getMaxSourcePartsSize(
                data.settings.max_replicated_merges_in_queue, merges_and_mutations_queued);

            if (max_source_parts_size > 0)
            {
                MergeTreeDataMergerMutator::FuturePart future_merged_part;
                if (merger_mutator.selectPartsToMerge(future_merged_part, false, max_source_parts_size, merge_pred))
                {
                    success = createLogEntryToMergeParts(zookeeper, future_merged_part.parts, future_merged_part.name, deduplicate);
                }
                else if (merge_pred.countMutations() > 0)
                {
                    /// Choose a part to mutate.

                    MergeTreeData::DataPartsVector data_parts = data.getDataPartsVector();
                    for (const auto & part : data_parts)
                    {
                        if (part->bytes_on_disk > max_source_parts_size)
                            continue;

                        std::optional<Int64> desired_mutation_version = merge_pred.getDesiredMutationVersion(part);
                        if (!desired_mutation_version)
                            continue;

                        if (createLogEntryToMutatePart(*part, *desired_mutation_version))
                        {
                            success = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    catch (...)
    {
        tryLogCurrentException(log, __PRETTY_FUNCTION__);
