  if (ctx != curCtx) {
    FOLLY_SDT(folly, request_context_switch_before, curCtx.get(), ctx.get());
    using std::swap;
    if (curCtx) {
      curCtx->onUnset();
    }
