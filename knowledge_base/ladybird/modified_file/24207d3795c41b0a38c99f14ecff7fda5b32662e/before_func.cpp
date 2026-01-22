    if (!candidate)
        return nullptr;
    return (*candidate)->range().contains(range) ? candidate->ptr() : nullptr;
}

Vector<Region*> AddressSpace::find_regions_intersecting(VirtualRange const& range)
{
    Vector<Region*> regions = {};
    size_t total_size_collected = 0;

    SpinlockLocker lock(m_lock);

    auto found_region = m_regions.find_largest_not_above(range.base().get());
    if (!found_region)
        return regions;
    for (auto iter = m_regions.begin_from((*found_region)->vaddr().get()); !iter.is_end(); ++iter) {
        if ((*iter)->range().base() < range.end() && (*iter)->range().end() > range.base()) {
            regions.append(*iter);

            total_size_collected += (*iter)->size() - (*iter)->range().intersect(range).size();
            if (total_size_collected == range.size())
                break;
