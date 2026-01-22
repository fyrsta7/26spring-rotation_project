void PageMemoryRights::on_btnDeselectall_clicked()
{
    QModelIndexList indexList = ui->pagetableWidget->selectionModel()->selectedIndexes();
    foreach(QModelIndex index, indexList)
    {
        ui->pagetableWidget->selectionModel()->select(index, QItemSelectionModel::Deselect);

    }
}
