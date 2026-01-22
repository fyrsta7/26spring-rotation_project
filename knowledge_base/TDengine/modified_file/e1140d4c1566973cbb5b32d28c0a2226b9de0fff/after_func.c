  return TSDB_CODE_SUCCESS;
}

static bool isIrowtsPseudoColumn(SExprInfo* pExprInfo) {
  char *name = pExprInfo->pExpr->_function.functionName;
  return (IS_TIMESTAMP_TYPE(pExprInfo->base.resSchema.type) && strcasecmp(name, "_irowts") == 0);
}

static bool isIsfilledPseudoColumn(SExprInfo* pExprInfo) {
  char *name = pExprInfo->pExpr->_function.functionName;
  return (IS_BOOLEAN_TYPE(pExprInfo->base.resSchema.type) && strcasecmp(name, "_isfilled") == 0);
}

static bool checkDuplicateTimestamps(STimeSliceOperatorInfo* pSliceInfo, SColumnInfoData* pTsCol,
                                     int32_t curIndex, int32_t rows) {


  int64_t currentTs = *(int64_t*)colDataGetData(pTsCol, curIndex);
  if (currentTs > pSliceInfo->win.ekey) {
    return false;
  }

  if ((pSliceInfo->prevTsSet == true) && (currentTs == pSliceInfo->prevTs)) {
    return true;
  }
