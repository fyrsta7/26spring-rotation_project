                              logAttrs(_nss),
                              "migrationSessionId"_attr = _migrationSessionId,
                              "fromShard"_attr = _fromShard);
                        _state = State::ReadyToCommit;
                    }
                }

                lastOpTimeWaited = lastResult.oplogTime;
            }
        }

        for (BSONArrayIteratorSorted oplogIter(oplogArray); oplogIter.more();) {
            auto oplogEntry = oplogIter.next().Obj();
            interruptBeforeProcessingPrePostImageOriginatingOp.executeIf(
                [&](const auto&) {
                    uasserted(6749200,
                              "Intentionally failing session migration before processing post/pre "
                              "image originating update oplog entry");
                },
                [&](const auto&) {
                    return !oplogEntry["needsRetryImage"].eoo() ||
                        !oplogEntry["preImageOpTime"].eoo() || !oplogEntry["postImageOpTime"].eoo();
                });
            try {
                lastResult =
                    _processSessionOplog(oplogEntry, lastResult, service, _cancellationToken);
            } catch (const ExceptionFor<ErrorCodes::TransactionTooOld>&) {
                // This means that the server has a newer txnNumber than the oplog being
                // migrated, so just skip it
                continue;
            }
        }
    }

    WriteConcernResult unusedWCResult;

    auto executor = Grid::get(service)->getExecutorPool()->getFixedExecutor();
    auto uniqueOpCtx =
        CancelableOperationContext(cc().makeOperationContext(), _cancellationToken, executor);
    uniqueOpCtx->setAlwaysInterruptAtStepDownOrUp_UNSAFE();

    uassertStatusOK(
        waitForWriteConcern(uniqueOpCtx.get(), lastResult.oplogTime, kMajorityWC, &unusedWCResult));

    {
        stdx::lock_guard<Latch> lk(_mutex);
        _state = State::Done;
    }
}

/**
 * Insert a new oplog entry by converting the oplogBSON into type 'n' oplog with the session
 * information. The new oplogEntry will also link to prePostImageTs if not null.
 */
SessionCatalogMigrationDestination::ProcessOplogResult
SessionCatalogMigrationDestination::_processSessionOplog(const BSONObj& oplogBSON,
                                                         const ProcessOplogResult& lastResult,
                                                         ServiceContext* serviceContext,
                                                         CancellationToken cancellationToken) {

    auto oplogEntry = parseOplog(oplogBSON);

    ProcessOplogResult result;
    result.sessionId = *oplogEntry.getSessionId();
    result.txnNum = *oplogEntry.getTxnNumber();

    if (oplogEntry.getOpType() == repl::OpTypeEnum::kNoop) {
        // Note: Oplog is already no-op type, no need to nest.
        // There are three types of type 'n' oplog format expected here:
        // (1) Oplog entries that has been transformed by a previous migration into a
        //     nested oplog. In this case, o field contains {$sessionMigrateInfo: 1}
        //     and o2 field contains the details of the original oplog.
        // (2) Oplog entries that contains the pre/post-image information of a
        //     findAndModify operation. In this case, o field contains the relevant info
        //     and o2 will be empty.
        // (3) Oplog entries that are a dead sentinel, which the donor sent over as the replacement
        //     for a prepare oplog entry or unprepared transaction commit oplog entry.
        // (4) Oplog entries that are a WouldChangeOwningShard sentinel entry, used for making
        //     retries of a WouldChangeOwningShard update or findAndModify fail with
        //     IncompleteTransactionHistory. In this case, the o field is non-empty and the o2
        //     field is an empty BSONObj.

        BSONObj object2;
        if (oplogEntry.getObject2()) {
            object2 = *oplogEntry.getObject2();
        } else {
            oplogEntry.setObject2(object2);
        }

        if (object2.isEmpty() && !isWouldChangeOwningShardSentinelOplogEntry(oplogEntry)) {
            result.isPrePostImage = true;

            uassert(40632,
                    str::stream() << "Can't handle 2 pre/post image oplog in a row. Prevoius oplog "
                                  << lastResult.oplogTime.getTimestamp().toString()
                                  << ", oplog ts: " << oplogEntry.getTimestamp().toString() << ": "
                                  << oplogBSON,
                    !lastResult.isPrePostImage);
        }
    } else {
        oplogEntry.setObject2(oplogBSON);  // TODO: strip redundant info?
    }

    const auto stmtIds = oplogEntry.getStatementIds();

    auto executor = Grid::get(serviceContext)->getExecutorPool()->getFixedExecutor();
    auto uniqueOpCtx =
        CancelableOperationContext(cc().makeOperationContext(), cancellationToken, executor);
    auto opCtx = uniqueOpCtx.get();
    opCtx->setAlwaysInterruptAtStepDownOrUp_UNSAFE();

    {
        auto lk = stdx::lock_guard(*opCtx->getClient());
        opCtx->setLogicalSessionId(result.sessionId);
        opCtx->setTxnNumber(result.txnNum);
    }

    // Irrespective of whether or not the oplog gets logged, we want to update the
    // entriesMigrated counter to signal that we have succesfully recieved the oplog
    // from the source and have processed it.
    _sessionOplogEntriesMigrated.addAndFetch(1);

    auto mongoDSessionCatalog = MongoDSessionCatalog::get(opCtx);
    auto ocs = mongoDSessionCatalog->checkOutSession(opCtx);

    auto txnParticipant = TransactionParticipant::get(opCtx);

    try {
        txnParticipant.beginOrContinue(opCtx,
                                       {result.txnNum},
                                       boost::none /* autocommit */,
                                       TransactionParticipant::TransactionActions::kNone);
        if (txnParticipant.checkStatementExecutedNoOplogEntryFetch(opCtx, stmtIds.front())) {
            // Skip the incoming statement because it has already been logged locally
            return lastResult;
        }
    } catch (const DBException& ex) {
        // If the transaction chain is incomplete because oplog was truncated, just ignore the
        // incoming oplog and don't attempt to 'patch up' the missing pieces.
        if (ex.code() == ErrorCodes::IncompleteTransactionHistory) {
            return lastResult;
        }

        if (stmtIds.front() == kIncompleteHistoryStmtId) {
            // No need to log entries for transactions whose history has been truncated
            invariant(stmtIds.size() == 1);
            return lastResult;
        }

        throw;
    }

    if (!result.isPrePostImage && !isWouldChangeOwningShardSentinelOplogEntry(oplogEntry)) {
        // Do not overwrite the "o" field if this is a pre/post image oplog entry. Also do not
        // overwrite it if this is a WouldChangeOwningShard sentinel oplog entry since it contains
        // a special BSONObj used for making retries fail with an IncompleteTransactionHistory
