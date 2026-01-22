      previewListModel->setData(previewListModel->index(row, PROGRESS), QVariant((double)fp[i]/h.filesize_at(i)));
      indexes << i;
    }
  }
  previewList->selectionModel()->select(previewListModel->index(0, NAME), QItemSelectionModel::Select);
  previewList->selectionModel()->select(previewListModel->index(0, SIZE), QItemSelectionModel::Select);
  previewList->selectionModel()->select(previewListModel->index(0, PROGRESS), QItemSelectionModel::Select);
  if (!previewListModel->rowCount()) {
    QMessageBox::critical(0, tr("Preview impossible"), tr("Sorry, we can't preview this file"));
    close();
  }
  connect(this, SIGNAL(readyToPreviewFile(QString)), parent, SLOT(previewFile(QString)));
  if (previewListModel->rowCount() == 1) {
    qDebug("Torrent file only contains one file, no need to display selection dialog before preview");
    // Only one file : no choice
    on_previewButton_clicked();
  }else{
    qDebug("Displaying media file selection dialog for preview");
    show();
  }
}

PreviewSelect::~PreviewSelect() {
