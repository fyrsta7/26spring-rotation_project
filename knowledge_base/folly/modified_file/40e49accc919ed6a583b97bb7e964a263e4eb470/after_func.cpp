    }
    staticCtx = newCtx;
    if (newCtx) {
      newCtx->onSet();
    }
  }
  return curCtx;
}

