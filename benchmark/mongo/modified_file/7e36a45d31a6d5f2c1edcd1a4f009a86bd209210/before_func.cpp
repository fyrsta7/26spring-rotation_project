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
                    updateEntry.setQ(BSON(CollectionType::kUuidFieldName << coll.getUuid()));
                    updateEntry.setU(mongo::write_ops::UpdateModification(
                        coll.toBSON(), write_ops::UpdateModification::ReplacementTag{}));
                    return updateEntry;
                }()});
                int collUpdateId = 1 + chunks.size() + 1;
                return txnClient.runCRUDOp(updateCollectionEntry, {collUpdateId});
            })
            .thenRunOn(txnExec)
            .then([&](const BatchedCommandResponse& updateCollectionEntryResponse) {
                uassertStatusOK(updateCollectionEntryResponse.toStatus());

                NamespacePlacementType placementInfo(
                    NamespaceString(coll.getNss()),
                    placementVersion.getTimestamp(),
                    std::vector<mongo::ShardId>(shardIds.cbegin(), shardIds.cend()));
                placementInfo.setUuid(coll.getUuid());

                write_ops::InsertCommandRequest insertPlacementEntry(
                    NamespaceString::kConfigsvrPlacementHistoryNamespace, {placementInfo.toBSON()});
                int historyUpdateId = 1 + chunks.size() + 2;
                return txnClient.runCRUDOp(insertPlacementEntry, {historyUpdateId});
            })
            .thenRunOn(txnExec)
            .then([](const BatchedCommandResponse& insertPlacementEntryResponse) {
                uassertStatusOK(insertPlacementEntryResponse.toStatus());
            })
            .semi();
    };

    // Ensure that this function will only return once the transaction gets majority committed
    auto wc = WriteConcernOptions{WriteConcernOptions::kMajority,
                                  WriteConcernOptions::SyncMode::UNSET,
                                  WriteConcernOptions::kNoTimeout};

    // This always runs in the shard role so should use a cluster transaction to guarantee targeting
    // the config server.
    bool useClusterTransaction = true;
    sharding_ddl_util::runTransactionOnShardingCatalog(
        opCtx, std::move(transactionChain), wc, osi, useClusterTransaction, executor);
}