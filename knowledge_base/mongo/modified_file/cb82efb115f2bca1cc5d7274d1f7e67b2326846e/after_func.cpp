
        // Use a ThreadPool to prefetch all the operations in a batch.
        prefetchOps(ops);
        
        std::vector< std::vector<BSONObj> > writerVectors(replWriterThreadCount);
        fillWriterVectors(ops, &writerVectors);
        LOG(2) << "replication batch size is " << ops.size() << endl;
        // We must grab this because we're going to grab write locks later.
        // We hold this mutex the entire time we're writing; it doesn't matter
        // because all readers are blocked anyway.
        SimpleMutex::scoped_lock fsynclk(filesLockedFsync);

        // stop all readers until we're done
        Lock::ParallelBatchWriterMode pbwm;

        ReplicationCoordinator* replCoord = getGlobalReplicationCoordinator();
        if (replCoord->getCurrentMemberState().primary() &&
            !replCoord->isWaitingForApplierToDrain()) {

            severe() << "attempting to replicate ops while primary";
            fassertFailed(28527);
        }

        applyOps(writerVectors);
        return applyOpsToOplog(&ops);
    }


    void SyncTail::fillWriterVectors(const std::deque<BSONObj>& ops,
                                     std::vector< std::vector<BSONObj> >* writerVectors) {

        for (std::deque<BSONObj>::const_iterator it = ops.begin();
             it != ops.end();
             ++it) {
            const BSONElement e = it->getField("ns");
