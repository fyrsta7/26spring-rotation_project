
//    std::cerr << "age: " << min_age << "\n";
//    std::cerr << "age_normalized: " << age_normalized << "\n";

    /// Map partition_size to 0..1
    double num_parts_normalized = mapPiecewiseLinearToUnit(partition_size, settings.min_parts_to_lower_base, settings.max_parts_to_lower_base);

//    std::cerr << "partition_size: " << partition_size << "\n";
//    std::cerr << "num_parts_normalized: " << num_parts_normalized << "\n";

    double combined_ratio = std::min(1.0, age_normalized + num_parts_normalized);

//    std::cerr << "combined_ratio: " << combined_ratio << "\n";

    double lowered_base = interpolateLinear(settings.base, 2.0, combined_ratio);

//    std::cerr << "------- lowered_base: " << lowered_base << "\n";

    return (sum_size + range_size * settings.size_fixed_cost_to_add) / (max_size + settings.size_fixed_cost_to_add) >= lowered_base;
}


void selectWithinPartition(
    const SimpleMergeSelector::PartsRange & parts,
    const size_t max_total_size_to_merge,
    Estimator & estimator,
    const SimpleMergeSelector::Settings & settings,
    double min_size_to_lower_base_log,
    double max_size_to_lower_base_log)
{
    size_t parts_count = parts.size();
    if (parts_count <= 1)
        return;

    for (size_t begin = 0; begin < parts_count; ++begin)
    {
        /// If too many parts, select only from first, to avoid complexity.
        if (begin > 1000)
            break;

        if (!parts[begin].shall_participate_in_merges)
            continue;

        size_t sum_size = parts[begin].size;
        size_t max_size = parts[begin].size;
        size_t min_age = parts[begin].age;

        for (size_t end = begin + 2; end <= parts_count; ++end)
        {
            assert(end > begin);
            if (settings.max_parts_to_merge_at_once && end - begin > settings.max_parts_to_merge_at_once)
                break;

            if (!parts[end - 1].shall_participate_in_merges)
