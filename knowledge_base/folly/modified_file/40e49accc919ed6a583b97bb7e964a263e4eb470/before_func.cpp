    }
    staticCtx = newCtx;
    if (newCtx) {
      newCtx->onSet();
    }
  }
  return curCtx;
}

std::shared_ptr<RequestContext>& RequestContext::getStaticContext() {
  using SingletonT = SingletonThreadLocal<std::shared_ptr<RequestContext>>;
  return SingletonT::get();
}

