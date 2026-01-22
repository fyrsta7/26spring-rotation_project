        done(ExecResult::Cancel);
    };
}

void LevelsDialog::revert_possible_changes()
{
    // FIXME: Find a faster way to revert all the changes that we have done.
    if (m_did_change && m_reference_bitmap) {
        for (int x = 0; x < m_reference_bitmap->width(); x++) {
            for (int y = 0; y < m_reference_bitmap->height(); y++) {
                m_editor->active_layer()->content_bitmap().set_pixel(x, y, m_reference_bitmap->get_pixel(x, y));
            }
        }
        m_editor->layers_did_change();
