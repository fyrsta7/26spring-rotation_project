
void UIManagerBinding::invalidate() const {
  uiManager_->setDelegate(nullptr);
}

jsi::Value UIManagerBinding::get(
    jsi::Runtime &runtime,
    jsi::PropNameID const &name) {
  auto methodName = name.utf8(runtime);
  SystraceSection s("UIManagerBinding::get", "name", methodName);

  // Convert shared_ptr<UIManager> to a raw ptr
  // Why? Because:
  // 1) UIManagerBinding strongly retains UIManager. The JS VM
  //    strongly retains UIManagerBinding (through the JSI).
  //    These functions are JSI functions and are only called via
  //    the JS VM; if the JS VM is torn down, those functions can't
  //    execute and these lambdas won't execute.
  // 2) The UIManager is only deallocated when all references to it
  //    are deallocated, including the UIManagerBinding. That only
  //    happens when the JS VM is deallocated. So, the raw pointer
  //    is safe.
  //
  // Even if it's safe, why not just use shared_ptr anyway as
  //  extra insurance?
  // 1) Using shared_ptr or weak_ptr when they're not needed is
  //    a pessimisation. It's more instructions executed without
  //    any additional value in this case.
  // 2) How and when exactly these lambdas is deallocated is
  //    complex. Adding shared_ptr to them which causes the UIManager
  //    to potentially live longer is unnecessary, complicated cognitive
  //    overhead.
  // 3) There is a strong suspicion that retaining UIManager from
  //    these C++ lambdas, which are retained by an object that is held onto
  //    by the JSI, caused some crashes upon deallocation of the
  //    Scheduler and JS VM. This could happen if, for instance, C++
  //    semantics cause these lambda to not be deallocated until
  //    a CPU tick (or more) after the JS VM is deallocated.
  UIManager *uiManager = uiManager_.get();

  // Semantic: Creates a new node with given pieces.
  if (methodName == "createNode") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        5,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto eventTarget =
              eventTargetFromValue(runtime, arguments[4], arguments[0]);
          if (!eventTarget) {
            react_native_assert(false);
            return jsi::Value::undefined();
          }
          return valueFromShadowNode(
              runtime,
              uiManager->createNode(
                  tagFromValue(arguments[0]),
                  stringFromValue(runtime, arguments[1]),
                  surfaceIdFromValue(runtime, arguments[2]),
                  RawProps(runtime, arguments[3]),
                  eventTarget));
        });
  }

  // Semantic: Clones the node with *same* props and *same* children.
  if (methodName == "cloneNode") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        1,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          return valueFromShadowNode(
              runtime,
              uiManager->cloneNode(
                  *shadowNodeFromValue(runtime, arguments[0])));
        });
  }

  if (methodName == "setIsJSResponder") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          uiManager->setIsJSResponder(
              shadowNodeFromValue(runtime, arguments[0]),
              arguments[1].getBool(),
              arguments[2].getBool());

          return jsi::Value::undefined();
        });
  }

  if (methodName == "findNodeAtPoint") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto node = shadowNodeFromValue(runtime, arguments[0]);
          auto locationX = (Float)arguments[1].getNumber();
          auto locationY = (Float)arguments[2].getNumber();
          auto onSuccessFunction =
              arguments[3].getObject(runtime).getFunction(runtime);
          auto targetNode =
              uiManager->findNodeAtPoint(node, Point{locationX, locationY});
          auto &eventTarget = targetNode->getEventEmitter()->eventTarget_;

          EventEmitter::DispatchMutex().lock();
          eventTarget->retain(runtime);
          auto instanceHandle = eventTarget->getInstanceHandle(runtime);
          eventTarget->release(runtime);
          EventEmitter::DispatchMutex().unlock();

          onSuccessFunction.call(runtime, std::move(instanceHandle));
          return jsi::Value::undefined();
        });
  }

  // Semantic: Clones the node with *same* props and *empty* children.
  if (methodName == "cloneNodeWithNewChildren") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        1,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          return valueFromShadowNode(
              runtime,
              uiManager->cloneNode(
                  *shadowNodeFromValue(runtime, arguments[0]),
                  ShadowNode::emptySharedShadowNodeSharedList()));
        });
  }

  // Semantic: Clones the node with *given* props and *same* children.
  if (methodName == "cloneNodeWithNewProps") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto const &rawProps = RawProps(runtime, arguments[1]);
          return valueFromShadowNode(
              runtime,
              uiManager->cloneNode(
                  *shadowNodeFromValue(runtime, arguments[0]),
                  nullptr,
                  &rawProps));
        });
  }

  // Semantic: Clones the node with *given* props and *empty* children.
  if (methodName == "cloneNodeWithNewChildrenAndProps") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto const &rawProps = RawProps(runtime, arguments[1]);
          return valueFromShadowNode(
              runtime,
              uiManager->cloneNode(
                  *shadowNodeFromValue(runtime, arguments[0]),
                  ShadowNode::emptySharedShadowNodeSharedList(),
                  &rawProps));
        });
  }

  if (methodName == "appendChild") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          uiManager->appendChild(
              shadowNodeFromValue(runtime, arguments[0]),
              shadowNodeFromValue(runtime, arguments[1]));
          return jsi::Value::undefined();
        });
  }

  if (methodName == "createChildSet") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        1,
        [](jsi::Runtime &runtime,
           jsi::Value const & /*thisValue*/,
           jsi::Value const * /*arguments*/,
           size_t /*count*/) noexcept -> jsi::Value {
          auto shadowNodeList = std::make_shared<ShadowNode::ListOfShared>(
              ShadowNode::ListOfShared({}));
          return valueFromShadowNodeList(runtime, shadowNodeList);
        });
  }

  if (methodName == "appendChildToSet") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [](jsi::Runtime &runtime,
           jsi::Value const & /*thisValue*/,
           jsi::Value const *arguments,
           size_t /*count*/) noexcept -> jsi::Value {
          auto shadowNodeList = shadowNodeListFromValue(runtime, arguments[0]);
          auto shadowNode = shadowNodeFromValue(runtime, arguments[1]);
          shadowNodeList->push_back(shadowNode);
          return jsi::Value::undefined();
        });
  }

  if (methodName == "completeRoot") {
    std::weak_ptr<UIManager> weakUIManager = uiManager_;
    // Enhanced version of the method that uses `backgroundExecutor` and
    // captures a shared pointer to `UIManager`.
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [weakUIManager, uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto runtimeSchedulerBinding =
              RuntimeSchedulerBinding::getBinding(runtime);
          auto surfaceId = surfaceIdFromValue(runtime, arguments[0]);

          if (!uiManager->backgroundExecutor_ ||
              (runtimeSchedulerBinding &&
               runtimeSchedulerBinding->getIsSynchronous())) {
            auto weakShadowNodeList =
                weakShadowNodeListFromValue(runtime, arguments[1]);
            auto shadowNodeList =
                shadowNodeListFromWeakList(weakShadowNodeList);
            if (shadowNodeList) {
              uiManager->completeSurface(surfaceId, shadowNodeList, {true});
            }
          } else {
            auto weakShadowNodeList =
                weakShadowNodeListFromValue(runtime, arguments[1]);
            static std::atomic_uint_fast8_t completeRootEventCounter{0};
            static std::atomic_uint_fast32_t mostRecentSurfaceId{0};
            completeRootEventCounter += 1;
            mostRecentSurfaceId = surfaceId;
            uiManager->backgroundExecutor_(
                [weakUIManager,
                 weakShadowNodeList,
                 surfaceId,
                 eventCount = completeRootEventCounter.load()] {
                  auto shouldYield = [=]() -> bool {
                    // If `completeRootEventCounter` was incremented, another
                    // `completeSurface` call has been scheduled and current
                    // `completeSurface` should yield to it.
                    return completeRootEventCounter > eventCount &&
                        mostRecentSurfaceId == surfaceId;
                  };
                  auto shadowNodeList =
                      shadowNodeListFromWeakList(weakShadowNodeList);
                  auto strongUIManager = weakUIManager.lock();
                  if (shadowNodeList && strongUIManager) {
                    strongUIManager->completeSurface(
                        surfaceId, shadowNodeList, {true, shouldYield});
                  }
                });
          }

          return jsi::Value::undefined();
        });
  }

  if (methodName == "registerEventHandler") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        1,
        [this](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto eventHandler =
              arguments[0].getObject(runtime).getFunction(runtime);
          eventHandler_ =
              std::make_unique<EventHandlerWrapper>(std::move(eventHandler));
          return jsi::Value::undefined();
        });
  }

  if (methodName == "getRelativeLayoutMetrics") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto layoutMetrics = uiManager->getRelativeLayoutMetrics(
              *shadowNodeFromValue(runtime, arguments[0]),
              shadowNodeFromValue(runtime, arguments[1]).get(),
              {/* .includeTransform = */ true});
          auto frame = layoutMetrics.frame;
          auto result = jsi::Object(runtime);
          result.setProperty(runtime, "left", frame.origin.x);
          result.setProperty(runtime, "top", frame.origin.y);
          result.setProperty(runtime, "width", frame.size.width);
          result.setProperty(runtime, "height", frame.size.height);
          return result;
        });
  }

  if (methodName == "dispatchCommand") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        3,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto shadowNode = shadowNodeFromValue(runtime, arguments[0]);
          if (shadowNode) {
            uiManager->dispatchCommand(
                shadowNode,
                stringFromValue(runtime, arguments[1]),
                commandArgsFromValue(runtime, arguments[2]));
          }
          return jsi::Value::undefined();
        });
  }

  if (methodName == "setNativeProps") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            const jsi::Value &,
            const jsi::Value *arguments,
            size_t) -> jsi::Value {
          uiManager->setNativeProps_DEPRECATED(
              shadowNodeFromValue(runtime, arguments[0]),
              RawProps(runtime, arguments[1]));

          return jsi::Value::undefined();
        });
  }

  // Legacy API
  if (methodName == "measureLayout") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        4,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto layoutMetrics = uiManager->getRelativeLayoutMetrics(
              *shadowNodeFromValue(runtime, arguments[0]),
              shadowNodeFromValue(runtime, arguments[1]).get(),
              {/* .includeTransform = */ false});

          if (layoutMetrics == EmptyLayoutMetrics) {
            auto onFailFunction =
                arguments[2].getObject(runtime).getFunction(runtime);
            onFailFunction.call(runtime);
            return jsi::Value::undefined();
          }

          auto onSuccessFunction =
              arguments[3].getObject(runtime).getFunction(runtime);
          auto frame = layoutMetrics.frame;

          onSuccessFunction.call(
              runtime,
              {jsi::Value{runtime, (double)frame.origin.x},
               jsi::Value{runtime, (double)frame.origin.y},
               jsi::Value{runtime, (double)frame.size.width},
               jsi::Value{runtime, (double)frame.size.height}});
          return jsi::Value::undefined();
        });
  }

  if (methodName == "measure") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto shadowNode = shadowNodeFromValue(runtime, arguments[0]);
          auto layoutMetrics = uiManager->getRelativeLayoutMetrics(
              *shadowNode, nullptr, {/* .includeTransform = */ true});
          auto onSuccessFunction =
              arguments[1].getObject(runtime).getFunction(runtime);

          if (layoutMetrics == EmptyLayoutMetrics) {
            onSuccessFunction.call(runtime, {0, 0, 0, 0, 0, 0});
            return jsi::Value::undefined();
          }
          auto newestCloneOfShadowNode =
              uiManager->getNewestCloneOfShadowNode(*shadowNode);

          auto layoutableShadowNode = traitCast<LayoutableShadowNode const *>(
              newestCloneOfShadowNode.get());
          Point originRelativeToParent = layoutableShadowNode != nullptr
              ? layoutableShadowNode->getLayoutMetrics().frame.origin
              : Point();

          auto frame = layoutMetrics.frame;
          onSuccessFunction.call(
              runtime,
              {jsi::Value{runtime, (double)originRelativeToParent.x},
               jsi::Value{runtime, (double)originRelativeToParent.y},
               jsi::Value{runtime, (double)frame.size.width},
               jsi::Value{runtime, (double)frame.size.height},
               jsi::Value{runtime, (double)frame.origin.x},
               jsi::Value{runtime, (double)frame.origin.y}});
          return jsi::Value::undefined();
        });
  }

  if (methodName == "measureInWindow") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          auto layoutMetrics = uiManager->getRelativeLayoutMetrics(
              *shadowNodeFromValue(runtime, arguments[0]),
              nullptr,
              {/* .includeTransform = */ true,
               /* includeViewportOffset = */ true});

          auto onSuccessFunction =
              arguments[1].getObject(runtime).getFunction(runtime);

          if (layoutMetrics == EmptyLayoutMetrics) {
            onSuccessFunction.call(runtime, {0, 0, 0, 0});
            return jsi::Value::undefined();
          }

          auto frame = layoutMetrics.frame;
          onSuccessFunction.call(
              runtime,
              {jsi::Value{runtime, (double)frame.origin.x},
               jsi::Value{runtime, (double)frame.origin.y},
               jsi::Value{runtime, (double)frame.size.width},
               jsi::Value{runtime, (double)frame.size.height}});
          return jsi::Value::undefined();
        });
  }

  if (methodName == "sendAccessibilityEvent") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        2,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          uiManager->sendAccessibilityEvent(
              shadowNodeFromValue(runtime, arguments[0]),
              stringFromValue(runtime, arguments[1]));

          return jsi::Value::undefined();
        });
  }

  if (methodName == "configureNextLayoutAnimation") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        3,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const & /*thisValue*/,
            jsi::Value const *arguments,
            size_t /*count*/) noexcept -> jsi::Value {
          uiManager->configureNextLayoutAnimation(
              runtime,
              // TODO: pass in JSI value instead of folly::dynamic to RawValue
              RawValue(commandArgsFromValue(runtime, arguments[0])),
              arguments[1],
              arguments[2]);
          return jsi::Value::undefined();
        });
  }

  if (methodName == "unstable_getCurrentEventPriority") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        0,
        [this](
            jsi::Runtime &,
            jsi::Value const &,
            jsi::Value const *,
            size_t) noexcept -> jsi::Value {
          return {serialize(currentEventPriority_)};
        });
  }

  if (methodName == "unstable_DefaultEventPriority") {
    return {serialize(ReactEventPriority::Default)};
  }

  if (methodName == "unstable_DiscreteEventPriority") {
    return {serialize(ReactEventPriority::Discrete)};
  }

  if (methodName == "findShadowNodeByTag_DEPRECATED") {
    return jsi::Function::createFromHostFunction(
        runtime,
        name,
        1,
        [uiManager](
            jsi::Runtime &runtime,
            jsi::Value const &,
            jsi::Value const *arguments,
            size_t) -> jsi::Value {
          auto shadowNode = uiManager->findShadowNodeByTag_DEPRECATED(
              tagFromValue(arguments[0]));

          if (!shadowNode) {
            return jsi::Value::null();
          }

          return valueFromShadowNode(runtime, shadowNode);
