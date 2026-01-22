    // correct
    DBDirectClient localClient(opCtx);
    return localClient.findOne(nss, BSONObj{}).isEmpty();
}

int getNumShards(OperationContext* opCtx) {
    const auto shardRegistry = Grid::get(opCtx)->shardRegistry();
    shardRegistry->reload(opCtx);

    return shardRegistry->getNumShards(opCtx);
}

void cleanupPartialChunksFromPreviousAttempt(OperationContext* opCtx,
                                             const UUID& uuid,
                                             const OperationSessionInfo& osi) {
    auto configShard = Grid::get(opCtx)->shardRegistry()->getConfigShard();

    // Remove the chunks matching uuid
    ConfigsvrRemoveChunks configsvrRemoveChunksCmd(uuid);
    configsvrRemoveChunksCmd.setDbName(DatabaseName::kAdmin);

    const auto swRemoveChunksResult = configShard->runCommandWithFixedRetryAttempts(
        opCtx,
        ReadPreferenceSetting{ReadPreference::PrimaryOnly},
        DatabaseName::kAdmin,
        CommandHelpers::appendMajorityWriteConcern(configsvrRemoveChunksCmd.toBSON(osi.toBSON())),
        Shard::RetryPolicy::kIdempotent);

    uassertStatusOKWithContext(
        Shard::CommandResponse::getEffectiveStatus(std::move(swRemoveChunksResult)),
        str::stream() << "Error removing chunks matching uuid " << uuid);
}

void updateCollectionMetadataInTransaction(OperationContext* opCtx,
                                           const std::shared_ptr<executor::TaskExecutor>& executor,
                                           const std::vector<ChunkType>& chunks,
                                           const CollectionType& coll,
                                           const ChunkVersion& placementVersion,
                                           const std::set<ShardId>& shardIds,
                                           const OperationSessionInfo& osi) {
    /*
     * As part of this chain, we will do the following operations:
     * 1. Delete any existing chunk entries (there can be 1 or 0 depending on whether we are
     * creating a collection or converting from unsplittable to splittable).
     * 2. Insert new chunk entries - there can be a maximum of  (2 * number of shards) or (number of
     * zones) new chunks.
     * 3. Replace the old collection entry with the new one (change the version and the shard key).
     * 4. Update the placement information.
     */
    const auto transactionChain = [&](const txn_api::TransactionClient& txnClient,
                                      ExecutorPtr txnExec) {
        write_ops::DeleteCommandRequest deleteOp(ChunkType::ConfigNS);
        deleteOp.setDeletes({[&] {
            write_ops::DeleteOpEntry entry;
            entry.setQ(BSON(ChunkType::collectionUUID.name() << coll.getUuid()));
            entry.setMulti(false);
            return entry;
        }()});

        return txnClient.runCRUDOp({deleteOp}, {0})
            .thenRunOn(txnExec)
            .then([&](const BatchedCommandResponse& deleteChunkEntryResponse) {
                uassertStatusOK(deleteChunkEntryResponse.toStatus());

                std::vector<StmtId> chunkStmts;
                BatchedCommandRequest insertChunkEntries([&]() {
                    write_ops::InsertCommandRequest insertOp(ChunkType::ConfigNS);
                    std::vector<BSONObj> entries;
                    entries.reserve(chunks.size());
                    chunkStmts.reserve(chunks.size());
                    int counter = 1;
                    for (const auto& chunk : chunks) {
                        entries.push_back(chunk.toConfigBSON());
                        chunkStmts.push_back({counter++});
                    }
                    insertOp.setDocuments(entries);
                    insertOp.setWriteCommandRequestBase([] {
                        write_ops::WriteCommandRequestBase wcb;
                        wcb.setOrdered(false);
                        return wcb;
                    }());
                    return insertOp;
                }());

                return txnClient.runCRUDOp(insertChunkEntries, chunkStmts);
            })
            .thenRunOn(txnExec)
            .then([&](const BatchedCommandResponse& insertChunkEntriesResponse) {
                uassertStatusOK(insertChunkEntriesResponse.toStatus());
                write_ops::UpdateCommandRequest updateCollectionEntry(CollectionType::ConfigNS);
                updateCollectionEntry.setUpdates({[&] {
                    write_ops::UpdateOpEntry updateEntry;
                    updateEntry.setMulti(false);
                    updateEntry.setUpsert(true);
