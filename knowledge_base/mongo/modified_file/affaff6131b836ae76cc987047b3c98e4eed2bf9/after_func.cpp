    if (qr.getResultFlags() & ResultFlag_ShardConfigStale) {
        BSONObj error;
        verify(peekError(&error));
        throw RecvStaleConfigException(
            (string) "stale config on lazy receive" + causedBy(getErrField(error)), error);
    }

    /* this assert would fire the way we currently work:
        verify( nReturned || cursorId == 0 );
    */
}

/** If true, safe to call next().  Requests more from server if necessary. */
bool DBClientCursor::more() {
    if (!_putBack.empty())
        return true;

    if (haveLimit && batch.pos >= nToReturn)
        return false;

    if (batch.pos < batch.nReturned)
        return true;

    if (cursorId == 0)
        return false;

