  while (true) {
    attempts++;

    auto status = tryCommit(transaction, commitOptions);
    if (status != CommitStatus::Failed) {
      return status;
    }

    // After multiple attempts, we failed to commit the transaction.
    // Something internally went terribly wrong.
    react_native_assert(attempts < 1024);
  }
}

CommitStatus ShadowTree::tryCommit(
    const ShadowTreeCommitTransaction &transaction,
    const CommitOptions &commitOptions) const {
  SystraceSection s("ShadowTree::tryCommit");

  auto telemetry = TransactionTelemetry{};
  telemetry.willCommit();

  CommitMode commitMode;
  auto oldRevision = ShadowTreeRevision{};
  auto newRevision = ShadowTreeRevision{};

  {
    // Reading `currentRevision_` in shared manner.
    std::shared_lock lock(commitMutex_);
    commitMode = commitMode_;
    oldRevision = currentRevision_;
  }

  auto const &oldRootShadowNode = oldRevision.rootShadowNode;
  auto newRootShadowNode = transaction(*oldRevision.rootShadowNode);

  if (!newRootShadowNode ||
      (commitOptions.shouldYield && commitOptions.shouldYield())) {
    return CommitStatus::Cancelled;
  }

  if (commitOptions.enableStateReconciliation) {
    auto updatedNewRootShadowNode =
        progressState(*newRootShadowNode, *oldRootShadowNode);
    if (updatedNewRootShadowNode) {
      newRootShadowNode =
          std::static_pointer_cast<RootShadowNode>(updatedNewRootShadowNode);
    }
  }

  // Layout nodes.
  std::vector<LayoutableShadowNode const *> affectedLayoutableNodes{};
  affectedLayoutableNodes.reserve(1024);

  telemetry.willLayout();
  telemetry.setAsThreadLocal();
  newRootShadowNode->layoutIfNeeded(&affectedLayoutableNodes);
  telemetry.unsetAsThreadLocal();
  telemetry.didLayout();

  // Seal the shadow node so it can no longer be mutated
  newRootShadowNode->sealRecursive();

  {
    // Updating `currentRevision_` in unique manner if it hasn't changed.
    std::unique_lock lock(commitMutex_);

    if (currentRevision_.number != oldRevision.number) {
      return CommitStatus::Failed;
    }

    auto newRevisionNumber = oldRevision.number + 1;

    newRootShadowNode = delegate_.shadowTreeWillCommit(
        *this, oldRootShadowNode, newRootShadowNode);

    if (!newRootShadowNode ||
        (commitOptions.shouldYield && commitOptions.shouldYield())) {
      return CommitStatus::Cancelled;
    }

    {
      std::lock_guard<std::mutex> dispatchLock(EventEmitter::DispatchMutex());

      updateMountedFlag(
          currentRevision_.rootShadowNode->getChildren(),
          newRootShadowNode->getChildren());
    }

    telemetry.didCommit();
    telemetry.setRevisionNumber(static_cast<int>(newRevisionNumber));

