        on_cell_data_change(cell, previous_data);
    did_update(UpdateFlag::DontInvalidateIndices);
}

void SheetModel::update()
