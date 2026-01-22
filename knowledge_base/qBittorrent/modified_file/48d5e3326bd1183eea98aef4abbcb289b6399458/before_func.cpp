#include <QDir>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QModelIndexList>
#include <QShortcut>
#include <QThread>
#include <QWheelEvent>

#include "base/bittorrent/torrentcontenthandler.h"
#include "base/path.h"
#include "base/utils/string.h"
#include "autoexpandabledialog.h"
#include "raisedmessagebox.h"
#include "torrentcontentfiltermodel.h"
#include "torrentcontentitemdelegate.h"
#include "torrentcontentmodel.h"
#include "torrentcontentmodelitem.h"
#include "uithememanager.h"
#include "utils.h"

#ifdef Q_OS_MACOS
#include "gui/macutilities.h"
#endif

TorrentContentWidget::TorrentContentWidget(QWidget *parent)
    : QTreeView(parent)
{
    setExpandsOnDoubleClick(false);
    setSortingEnabled(true);
    header()->setSortIndicator(0, Qt::AscendingOrder);
    header()->setFirstSectionMovable(true);
    header()->setContextMenuPolicy(Qt::CustomContextMenu);

    m_model = new TorrentContentModel(this);
    connect(m_model, &TorrentContentModel::renameFailed, this, [this](const QString &errorMessage)
    {
