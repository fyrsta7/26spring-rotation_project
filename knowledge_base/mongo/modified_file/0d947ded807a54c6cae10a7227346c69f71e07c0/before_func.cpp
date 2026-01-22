                cc->mayUpgradeStorage();
                cc->storeOpForSlave( last );
                exhaust = cc->queryOptions() & QueryOption_Exhaust;
            }
        }

        QueryResult *qr = (QueryResult *) b.buf();
        qr->len = b.len();
        qr->setOperation(opReply);
        qr->_resultFlags() = resultFlags;
        qr->cursorId = cursorid;
        qr->startingFrom = start;
        qr->nReturned = n;
        b.decouple();

        return qr;
    }

    /* { count: "collectionname"[, query: <query>] }
       returns -1 on ns does not exist error.
    */
    long long runCount( const char *ns, const BSONObj &cmd, string &err ) {
        Client::Context cx(ns);
        NamespaceDetails *d = nsdetails( ns );
        if ( !d ) {
            err = "ns missing";
            return -1;
        }
        BSONObj query = cmd.getObjectField("query");

        // count of all objects
        if ( query.isEmpty() ) {
            return applySkipLimit( d->stats.nrecords , cmd );
        }
        
        string exceptionInfo;
        long long count = 0;
        long long skip = cmd["skip"].numberLong();
        long long limit = cmd["limit"].numberLong();
        bool simpleEqualityMatch;
        shared_ptr<Cursor> cursor = NamespaceDetailsTransient::getCursor( ns, query, BSONObj(), false, &simpleEqualityMatch );
        ClientCursor::CleanupPointer ccPointer;
        ccPointer.reset( new ClientCursor( QueryOption_NoCursorTimeout, cursor, ns ) );
        try {
            while( cursor->ok() ) {
                if ( !ccPointer->yieldSometimes( simpleEqualityMatch ? ClientCursor::DontNeed : ClientCursor::MaybeCovered ) ||
                    !cursor->ok() ) {
                    break;
                }

                // With simple equality matching there is no need to use the matcher because the bounds
                // are enforced by the FieldRangeVectorIterator and only key fields have constraints.  There
                // is no need do key deduping because an exact value is specified in the query for all key
                // fields and duplicate keys are not allowed per document.
                // NOTE In the distant past we used a min/max bounded BtreeCursor with a shallow
                // equality comparison to check for matches in the simple match case.  That may be
                // more performant, but I don't think we've measured the performance.
                if ( simpleEqualityMatch ||
                    ( ( !cursor->matcher() || cursor->matcher()->matchesCurrent( cursor.get() ) ) &&
                        !cursor->getsetdup( cursor->currLoc() ) ) ) {
                        
                    if ( skip > 0 ) {
                        --skip;
                    }
                    else {
                        ++count;
                        if ( limit > 0 && count >= limit ) {
