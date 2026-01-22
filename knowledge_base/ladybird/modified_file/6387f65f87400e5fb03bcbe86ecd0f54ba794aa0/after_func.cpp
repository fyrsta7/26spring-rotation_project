{
    return section_data(section).size;
}

HeaderView::SectionData& HeaderView::section_data(int section) const
{
    VERIFY(model());
    if (static_cast<size_t>(section) >= m_section_data.size()) {
