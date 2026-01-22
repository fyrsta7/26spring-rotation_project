
  return pList;
}

int32_t dumpQueryTableCond(const SQueryTableDataCond* src, SQueryTableDataCond* dst) {
  memcpy((void*)dst, (void*)src, sizeof(SQueryTableDataCond));
  dst->colList = taosMemoryCalloc(src->numOfCols, sizeof(SColumnInfo));
  for (int i = 0; i < src->numOfCols; i++) {
    dst->colList[i] = src->colList[i];
  }
  return 0;
}

int32_t startGroupTableMergeScan(SOperatorInfo* pOperator) {
  STableMergeScanInfo* pInfo = pOperator->info;
  SExecTaskInfo*       pTaskInfo = pOperator->pTaskInfo;

  {
    size_t  numOfTables = tableListGetSize(pInfo->base.pTableListInfo);
    int32_t i = pInfo->tableStartIndex + 1;
    for (; i < numOfTables; ++i) {
      STableKeyInfo* tableKeyInfo = tableListGetInfo(pInfo->base.pTableListInfo, i);
      if (tableKeyInfo->groupId != pInfo->groupId) {
        break;
      }
    }
    pInfo->tableEndIndex = i - 1;
  }

  int32_t tableStartIdx = pInfo->tableStartIndex;
  int32_t tableEndIdx = pInfo->tableEndIndex;

  pInfo->base.dataReader = NULL;

  // todo the total available buffer should be determined by total capacity of buffer of this task.
  // the additional one is reserved for merge result
  // pInfo->sortBufSize = pInfo->bufPageSize * (tableEndIdx - tableStartIdx + 1 + 1);
  int32_t kWay = (TSDB_MAX_BYTES_PER_ROW * 2) / (pInfo->pResBlock->info.rowSize);
  if (kWay >= 256) {
    kWay = 256;
  } else if (kWay <= 2) {
    kWay = 2;
  } else {
    int i = 2; 
    while (i * 2 <= kWay) i = i * 2;
    kWay = i;
  }

  pInfo->sortBufSize = pInfo->bufPageSize * (kWay + 1);
  int32_t numOfBufPage = pInfo->sortBufSize / pInfo->bufPageSize;
  pInfo->pSortHandle = tsortCreateSortHandle(pInfo->pSortInfo, SORT_MULTISOURCE_MERGE, pInfo->bufPageSize, numOfBufPage,
                                             pInfo->pSortInputBlock, pTaskInfo->id.str);

  tsortSetFetchRawDataFp(pInfo->pSortHandle, getTableDataBlockImpl, NULL, NULL);

  // one table has one data block
  int32_t numOfTable = tableEndIdx - tableStartIdx + 1;
  pInfo->queryConds = taosArrayInit(numOfTable, sizeof(SQueryTableDataCond));

  for (int32_t i = 0; i < numOfTable; ++i) {
    STableMergeScanSortSourceParam param = {0};
    param.readerIdx = i;
    param.pOperator = pOperator;
    param.inputBlock = createOneDataBlock(pInfo->pResBlock, false);

    taosArrayPush(pInfo->sortSourceParams, &param);

    SQueryTableDataCond cond;
    dumpQueryTableCond(&pInfo->base.cond, &cond);
    taosArrayPush(pInfo->queryConds, &cond);
  }

  for (int32_t i = 0; i < numOfTable; ++i) {
    SSortSource*                    ps = taosMemoryCalloc(1, sizeof(SSortSource));
    STableMergeScanSortSourceParam* param = taosArrayGet(pInfo->sortSourceParams, i);
