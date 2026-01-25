void PreviewSelect::on_previewButton_clicked() {
  QModelIndex index;
  QModelIndexList selectedIndexes = previewList->selectionModel()->selectedRows(NAME);
  if (selectedIndexes.size() == 0) return;
  // Flush data
  h.flush_cache();

  QString path;
  foreach (index, selectedIndexes) {
    path = h.absolute_files_path().at(indexes.at(index.row()));
    // File
    if (QFile::exists(path)) {
      emit readyToPreviewFile(path);
    } else {
      QMessageBox::critical(0, tr("Preview impossible"), tr("Sorry, we can't preview this file"));
    }
    close();
    return;
  }
  qDebug("Cannot find file: %s", path.toLocal8Bit().data());
  QMessageBox::critical(0, tr("Preview impossible"), tr("Sorry, we can't preview this file"));
  close();
}