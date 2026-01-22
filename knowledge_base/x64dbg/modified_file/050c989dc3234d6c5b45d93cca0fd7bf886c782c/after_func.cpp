void PageMemoryRights::on_btnDeselectall_clicked()
{
    QModelIndexList selectedIndexes = ui->pagetableWidget->selectionModel()->selectedIndexes();
    if(selectedIndexes.isEmpty())
        return;

    QModelIndex topLeft = selectedIndexes.first();
    QModelIndex bottomRight = selectedIndexes.last();

    QItemSelection selection(topLeft, bottomRight);
    ui->pagetableWidget->selectionModel()->select(selection, QItemSelectionModel::Deselect);
}
