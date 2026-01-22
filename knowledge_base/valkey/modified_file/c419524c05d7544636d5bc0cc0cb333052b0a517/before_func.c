         * as an interaction, since we always send REPLCONF ACK commands
         * that take some time to just fill the socket output buffer.
         * We just rely on data / pings received for timeout detection. */
        if (!c->flag.primary) c->last_interaction = server.unixtime;
    }
    if (!clientHasPendingReplies(c)) {
        c->sentlen = 0;
        if (connHasWriteHandler(c->conn)) {
            connSetWriteHandler(c->conn, NULL);
        }

        /* Close connection after entire reply has been sent. */
        if (c->flag.close_after_reply) {
            freeClientAsync(c);
            return C_ERR;
        }
    }
    /* Update client's memory usage after writing.*/
    updateClientMemUsageAndBucket(c);
    return C_OK;
}

/* Write data in output buffers to client. Return C_OK if the client
 * is still valid after the call, C_ERR if it was freed because of some
 * error.
 *
 * This function is called by main-thread only */
int writeToClient(client *c) {
    if (c->io_write_state != CLIENT_IDLE || c->io_read_state != CLIENT_IDLE) return C_OK;

    c->nwritten = 0;
    c->write_flags = 0;

    if (getClientType(c) == CLIENT_TYPE_REPLICA) {
        writeToReplica(c);
    } else {
        _writeToClient(c);
    }

    return postWriteToClient(c);
}

/* Write event handler. Just send data to the client. */
void sendReplyToClient(connection *conn) {
    client *c = connGetPrivateData(conn);
    if (trySendWriteToIOThreads(c) == C_OK) return;
    writeToClient(c);
}

void handleQbLimitReached(client *c) {
    sds ci = catClientInfoString(sdsempty(), c, server.hide_user_data_from_log), bytes = sdsempty();
    bytes = sdscatrepr(bytes, c->querybuf, 64);
    serverLog(LL_WARNING, "Closing client that reached max query buffer length: %s (qbuf initial bytes: %s)", ci,
              bytes);
    sdsfree(ci);
    sdsfree(bytes);
    freeClientAsync(c);
    server.stat_client_qbuf_limit_disconnections++;
}

/* Handle read errors and update statistics.
 *
 * Called only from the main thread.
 * If the read was done in an I/O thread, this function is invoked after the
 * read job has completed, in the main thread context.
