}

void RangeAllocator::dump() const
{
    VERIFY(m_lock.is_locked());
    dbgln("RangeAllocator({})", this);
    for (auto& range : m_available_ranges) {
        dbgln("    {:x} -> {:x}", range.base().get(), range.end().get() - 1);
    }
}

void RangeAllocator::carve_at_index(int index, const Range& range)
{
    VERIFY(m_lock.is_locked());
    auto remaining_parts = m_available_ranges[index].carve(range);
    VERIFY(remaining_parts.size() >= 1);
    VERIFY(m_total_range.contains(remaining_parts[0]));
    m_available_ranges[index] = remaining_parts[0];
    if (remaining_parts.size() == 2) {
        VERIFY(m_total_range.contains(remaining_parts[1]));
        m_available_ranges.insert(index + 1, move(remaining_parts[1]));
    }
}
