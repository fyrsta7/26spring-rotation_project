void PageMemoryRights::on_btnSelectall_clicked()
{
    for(int i = 0; i < ui->pagetableWidget->rowCount(); i++)
    {
        for(int j = 0; j < ui->pagetableWidget->columnCount(); j++)
        {
            QModelIndex idx = (ui->pagetableWidget->model()->index(i, j));
            ui->pagetableWidget->selectionModel()->select(idx, QItemSelectionModel::Select);
        }
    }
}
