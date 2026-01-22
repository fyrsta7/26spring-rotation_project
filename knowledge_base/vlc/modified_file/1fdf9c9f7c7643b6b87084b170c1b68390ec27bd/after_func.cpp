#include "mlplaylistmedia.hpp"

#include "playlist/playlist_controller.hpp"
#include "playlist/media.hpp"

//=================================================================================================
// MLPlaylistModel
//=================================================================================================

/* explicit */ MLPlaylistModel::MLPlaylistModel(QObject * parent)
    : MLBaseModel(parent) {}

//-------------------------------------------------------------------------------------------------
// Interface
//-------------------------------------------------------------------------------------------------

/* Q_INVOKABLE */ void MLPlaylistModel::insert(const QVariantList & items, int at)
{
    assert(m_mediaLib);

    int64_t id = parentId().id;

    assert(id);

    if (unlikely(m_transactionPending))
        return;

    QVector<vlc::playlist::Media> medias = vlc::playlist::toMediaList(items);

    setTransactionPending(true);

    struct Ctx {
        std::vector<std::unique_ptr<MLItem>> medias;
    };
    m_mediaLib->runOnMLThread<Ctx>(this,
    //ML thread
    [medias, id, at](vlc_medialibrary_t* ml, Ctx& ctx) {
        std::vector<int64_t> mediaIdList;
        for (const auto& media : medias)
        {
            assert(media.raw());

            const char * const uri = media.raw()->psz_uri;

            vlc_ml_media_t * ml_media = vlc_ml_get_media_by_mrl(ml, uri);

            if (ml_media == nullptr)
            {
                ml_media = vlc_ml_new_external_media(ml, uri);
                if (ml_media == nullptr)
                    continue;
            }
