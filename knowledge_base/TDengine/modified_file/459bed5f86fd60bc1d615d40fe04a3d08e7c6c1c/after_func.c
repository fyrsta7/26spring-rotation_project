  }
}

static void doSaveCurrentVal(SqlFunctionCtx* pCtx, int32_t rowIndex, int64_t currentTs, int32_t type, char* pData) {
  SResultRowEntryInfo* pResInfo = GET_RES_INFO(pCtx);
  SFirstLastRes*       pInfo = GET_ROWCELL_INTERBUF(pResInfo);

  if (IS_VAR_DATA_TYPE(type)) {
    pInfo->bytes = varDataTLen(pData);
  }

  memcpy(pInfo->buf, pData, pInfo->bytes);
  pInfo->ts = currentTs;
  firstlastSaveTupleData(pCtx->pSrcBlock, rowIndex, pCtx, pInfo);

  pInfo->hasResult = true;
}

// This ordinary first function does not care if current scan is ascending order or descending order scan
// the OPTIMIZED version of first function will only handle the ascending order scan
int32_t firstFunction(SqlFunctionCtx* pCtx) {
  int32_t numOfElems = 0;

  SResultRowEntryInfo* pResInfo = GET_RES_INFO(pCtx);
  SFirstLastRes*       pInfo = GET_ROWCELL_INTERBUF(pResInfo);

  SInputColumnInfoData* pInput = &pCtx->input;
  SColumnInfoData*      pInputCol = pInput->pData[0];

  pInfo->bytes = pInputCol->info.bytes;

  // All null data column, return directly.
  if (pInput->colDataAggIsSet && (pInput->pColumnDataAgg[0]->numOfNull == pInput->totalRows)) {
    ASSERT(pInputCol->hasNull == true);
    // save selectivity value for column consisted of all null values
    firstlastSaveTupleData(pCtx->pSrcBlock, pInput->startRowIndex, pCtx, pInfo);
    return 0;
  }

  SColumnDataAgg* pColAgg = (pInput->colDataAggIsSet) ? pInput->pColumnDataAgg[0] : NULL;

  TSKEY startKey = getRowPTs(pInput->pPTS, 0);
  TSKEY endKey = getRowPTs(pInput->pPTS, pInput->totalRows - 1);

  int32_t blockDataOrder = (startKey <= endKey) ? TSDB_ORDER_ASC : TSDB_ORDER_DESC;

  //  please ref. to the comment in lastRowFunction for the reason why disabling the opt version of last/first function.
  //  we will use this opt implementation in an new version that is only available in scan subplan
#if 0
  if (blockDataOrder == TSDB_ORDER_ASC) {
    // filter according to current result firstly
    if (pResInfo->numOfRes > 0) {
      if (pInfo->ts < startKey) {
        return TSDB_CODE_SUCCESS;
      }
    }

    for (int32_t i = pInput->startRowIndex; i < pInput->startRowIndex + pInput->numOfRows; ++i) {
      if (pInputCol->hasNull && colDataIsNull(pInputCol, pInput->totalRows, i, pColAgg)) {
        continue;
      }

      numOfElems++;

      char* data = colDataGetData(pInputCol, i);
      TSKEY cts = getRowPTs(pInput->pPTS, i);
      if (pResInfo->numOfRes == 0 || pInfo->ts > cts) {
        doSaveCurrentVal(pCtx, i, cts, pInputCol->info.type, data);
        break;
      }
    }
  } else {
    // in case of descending order time stamp serial, which usually happens as the results of the nest query,
    // all data needs to be check.
    if (pResInfo->numOfRes > 0) {
      if (pInfo->ts < endKey) {
        return TSDB_CODE_SUCCESS;
      }
    }

    for (int32_t i = pInput->numOfRows + pInput->startRowIndex - 1; i >= pInput->startRowIndex; --i) {
      if (pInputCol->hasNull && colDataIsNull(pInputCol, pInput->totalRows, i, pColAgg)) {
        continue;
      }

      numOfElems++;

      char* data = colDataGetData(pInputCol, i);
      TSKEY cts = getRowPTs(pInput->pPTS, i);

      if (pResInfo->numOfRes == 0 || pInfo->ts > cts) {
        doSaveCurrentVal(pCtx, i, cts, pInputCol->info.type, data);
        break;
      }
    }
  }
#else
  int64_t* pts = (int64_t*) pInput->pPTS->pData;
  for (int32_t i = pInput->startRowIndex; i < pInput->startRowIndex + pInput->numOfRows; ++i) {
    if (pInputCol->hasNull && colDataIsNull(pInputCol, pInput->totalRows, i, pColAgg)) {
      continue;
