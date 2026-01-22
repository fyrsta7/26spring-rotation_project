        done(ExecResult::Cancel);
    };
}

void LevelsDialog::revert_possible_changes()
{
    if (m_did_change && m_reference_bitmap) {
        MUST(m_editor->active_layer()->set_bitmaps(m_reference_bitmap.release_nonnull(), m_editor->active_layer()->mask_bitmap()));
