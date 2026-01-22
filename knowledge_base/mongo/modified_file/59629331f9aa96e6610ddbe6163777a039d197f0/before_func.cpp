        nsToDatabase(ns, db);

        if ( *ns == '.' || *ns == 0 ) {
            log() << "replSet skipping bad op in oplog: " << o.toString() << endl;
            return;
        }

        Client::Context ctx(ns);
        ctx.getClient()->curop()->reset();

        /* todo : if this asserts, do we want to ignore or not? */
        applyOperation_inlock(o);
    }

    bool ReplSetImpl::initialSyncOplogApplication(
        string hn, 
        const Member *primary,
        OpTime applyGTE,
        OpTime minValid)
    { 
        if( primary == 0 ) return false;
        OpTime ts;
        try {
            OplogReader r;
            if( !r.connect(hn) ) { 
                log(2) << "replSet can't connect to " << hn << " to read operations" << rsLog;
                return false;
            }

            r.query(rsoplog, bo());
            assert( r.haveCursor() );

            while( 1 ) { 
                if( !r.more() )
                    break;
                BSONObj o = r.nextSafe(); /* note we might get "not master" at some point */
                {
                    writelock lk("");

                    ts = o["ts"]._opTime();

                    /* if we have become primary, we dont' want to apply things from elsewhere
                        anymore. assumePrimary is in the db lock so we are safe as long as 
                        we check after we locked above. */
                    if( box.getPrimary() != primary ) {
                        throw DBException("primary changed",0);
                    }

                    if( ts >= applyGTE ) {
