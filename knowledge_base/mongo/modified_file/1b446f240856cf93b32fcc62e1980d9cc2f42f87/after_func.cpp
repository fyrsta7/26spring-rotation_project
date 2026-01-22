
    namespace {
        int mdb_handle_error(WT_EVENT_HANDLER *handler, WT_SESSION *session,
                             int errorCode, const char *message) {
            error() << "WiredTiger (" << errorCode << ") " << message;
            return 0;
        }

        int mdb_handle_message( WT_EVENT_HANDLER *handler, WT_SESSION *session,
                                const char *message) {
            log() << "WiredTiger " << message;
            return 0;
        }

        int mdb_handle_progress( WT_EVENT_HANDLER *handler, WT_SESSION *session,
                                 const char *operation, uint64_t progress) {
            log() << "WiredTiger progress " << operation << " " << progress;
            return 0;
        }

        int mdb_handle_close( WT_EVENT_HANDLER *handler, WT_SESSION *session,
                              WT_CURSOR *cursor) {
            return 0;
        }

    }

    WiredTigerKVEngine::WiredTigerKVEngine( const std::string& path,
                                            const std::string& extraOpenOptions,
                                            bool durable )
        : _durable( durable ),
          _epoch( 0 ),
          _sizeStorerSyncTracker( 100000, 60 * 1000 ) {

        _eventHandler.handle_error = mdb_handle_error;
        _eventHandler.handle_message = mdb_handle_message;
        _eventHandler.handle_progress = mdb_handle_progress;
        _eventHandler.handle_close = mdb_handle_close;

        int cacheSizeGB = 1;

        {
            ProcessInfo pi;
            unsigned long long memSizeMB  = pi.getMemSizeMB();
            if ( memSizeMB  > 0 ) {
                double cacheMB = memSizeMB / 50;
                cacheSizeGB = static_cast<int>( cacheMB / 1024 );
                if ( cacheSizeGB < 1 )
                    cacheSizeGB = 1;
            }
        }

        if ( _durable ) {
            boost::filesystem::path journalPath = path;
            journalPath /= "journal";
            if ( !boost::filesystem::exists( journalPath ) ) {
                try {
                    boost::filesystem::create_directory( journalPath );
                }
                catch( std::exception& e) {
                    log() << "error creating journal dir " << journalPath.string() << ' ' << e.what();
                    throw;
                }
