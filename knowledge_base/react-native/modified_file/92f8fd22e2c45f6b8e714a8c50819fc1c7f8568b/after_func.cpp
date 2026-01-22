void Binding::schedulerDidFinishTransaction(
    MountingCoordinator::Shared const &mountingCoordinator) {
  SystraceSection s("FabricUIManager::schedulerDidFinishTransaction");

  auto mountingTransaction = mountingCoordinator->pullTransaction();

  if (!mountingTransaction.has_value()) {
    return;
  }

  auto telemetry = mountingTransaction->getTelemetry();
  auto surfaceId = mountingTransaction->getSurfaceId();
  auto &mutations = mountingTransaction->getMutations();

  std::vector<local_ref<jobject>> queue;
  // Upper bound estimation of mount items to be delivered to Java side.
  int size = mutations.size() * 3 + 42;

  long finishTransactionStartTime = getTime();

  local_ref<JArrayClass<JMountItem::javaobject>> mountItemsArray =
      JArrayClass<JMountItem::javaobject>::newArray(size);

  auto mountItems = *(mountItemsArray);
  std::unordered_set<Tag> deletedViewTags;

  int position = 0;
  for (const auto &mutation : mutations) {
    auto oldChildShadowView = mutation.oldChildShadowView;
    auto newChildShadowView = mutation.newChildShadowView;

    bool isVirtual = newChildShadowView.layoutMetrics == EmptyLayoutMetrics &&
        oldChildShadowView.layoutMetrics == EmptyLayoutMetrics;

    switch (mutation.type) {
      case ShadowViewMutation::Create: {
        if (mutation.newChildShadowView.props->revision > 1 ||
            deletedViewTags.find(mutation.newChildShadowView.tag) !=
                deletedViewTags.end()) {
          mountItems[position++] =
              createCreateMountItem(javaUIManager_, mutation, surfaceId);
        }
        break;
      }
      case ShadowViewMutation::Remove: {
        if (!isVirtual) {
          mountItems[position++] =
              createRemoveMountItem(javaUIManager_, mutation);
        }
        break;
      }
      case ShadowViewMutation::Delete: {
        mountItems[position++] =
            createDeleteMountItem(javaUIManager_, mutation);

        deletedViewTags.insert(mutation.oldChildShadowView.tag);
        break;
      }
      case ShadowViewMutation::Update: {
        if (!isVirtual) {
          if (mutation.oldChildShadowView.props !=
              mutation.newChildShadowView.props) {
            mountItems[position++] =
                createUpdatePropsMountItem(javaUIManager_, mutation);
          }
          if (mutation.oldChildShadowView.localData !=
              mutation.newChildShadowView.localData) {
            mountItems[position++] =
                createUpdateLocalData(javaUIManager_, mutation);
          }
          if (mutation.oldChildShadowView.state !=
              mutation.newChildShadowView.state) {
            mountItems[position++] =
                createUpdateStateMountItem(javaUIManager_, mutation);
          }

          auto updateLayoutMountItem =
              createUpdateLayoutMountItem(javaUIManager_, mutation);
          if (updateLayoutMountItem) {
            mountItems[position++] = updateLayoutMountItem;
          }
        }

        if (mutation.oldChildShadowView.eventEmitter !=
            mutation.newChildShadowView.eventEmitter) {
          auto updateEventEmitterMountItem =
              createUpdateEventEmitterMountItem(javaUIManager_, mutation);
          if (updateEventEmitterMountItem) {
            mountItems[position++] = updateEventEmitterMountItem;
          }
        }
        break;
      }
      case ShadowViewMutation::Insert: {
        if (!isVirtual) {
          // Insert item
          mountItems[position++] =
              createInsertMountItem(javaUIManager_, mutation);

          if (mutation.newChildShadowView.props->revision > 1 ||
              deletedViewTags.find(mutation.newChildShadowView.tag) !=
                  deletedViewTags.end()) {
            mountItems[position++] =
                createUpdatePropsMountItem(javaUIManager_, mutation);

            // State
            if (mutation.newChildShadowView.state) {
              mountItems[position++] =
                  createUpdateStateMountItem(javaUIManager_, mutation);
            }
          }

          // LocalData
          if (mutation.newChildShadowView.localData) {
            mountItems[position++] =
                createUpdateLocalData(javaUIManager_, mutation);
          }

          // Layout
          auto updateLayoutMountItem =
              createUpdateLayoutMountItem(javaUIManager_, mutation);
          if (updateLayoutMountItem) {
            mountItems[position++] = updateLayoutMountItem;
          }
        }

        // EventEmitter
        auto updateEventEmitterMountItem =
            createUpdateEventEmitterMountItem(javaUIManager_, mutation);
        if (updateEventEmitterMountItem) {
          mountItems[position++] = updateEventEmitterMountItem;
        }

        break;
      }
      default: {
        break;
      }
    }
  }

  static auto createMountItemsBatchContainer =
      jni::findClassStatic(UIManagerJavaDescriptor)
          ->getMethod<alias_ref<JMountItem>(
              jtypeArray<JMountItem::javaobject>, jint)>(
              "createBatchMountItem");

  auto batch = createMountItemsBatchContainer(
      javaUIManager_, mountItemsArray.get(), position);

  static auto scheduleMountItems =
      jni::findClassStatic(UIManagerJavaDescriptor)
          ->getMethod<void(JMountItem::javaobject, jlong, jlong, jlong, jlong)>(
              "scheduleMountItems");

  long finishTransactionEndTime = getTime();

  scheduleMountItems(
      javaUIManager_,
      batch.get(),
      telemetry.getCommitStartTime(),
      telemetry.getLayoutTime(),
      finishTransactionStartTime,
      finishTransactionEndTime);
}
