void PageMemoryRights::on_btnSelectall_clicked()
{
    const auto rowCount = ui->pagetableWidget->rowCount();
    const auto columnCount = ui->pagetableWidget->columnCount();

    QModelIndex topLeft = ui->pagetableWidget->model()->index(0, 0);
    QModelIndex bottomRight = ui->pagetableWidget->model()->index(rowCount - 1, columnCount - 1);

    QItemSelection selection(topLeft, bottomRight);
    ui->pagetableWidget->selectionModel()->select(selection, QItemSelectionModel::Select);
}
