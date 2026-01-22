    code = tBlockDataAppendBlockRow(pBlockData, pRow->pBlockData, pRow->iRow);
    if (code) goto _err;
  } else {
    ASSERT(0);
  }
  pBlockData->nRow++;

  return code;

_err:
  return code;
}

void tBlockDataGetColData(SBlockData *pBlockData, int16_t cid, SColData **ppColData) {
  ASSERT(cid != PRIMARYKEY_TIMESTAMP_COL_ID);
  int32_t lidx = 0;
  int32_t ridx = pBlockData->nColData - 1;

  while (lidx <= ridx) {
    int32_t   midx = (lidx + ridx) >> 2;
    SColData *pColData = tBlockDataGetColDataByIdx(pBlockData, midx);
    int32_t   c = (pColData->cid == cid) ? 0 : ((pColData->cid > cid) ? 1 : -1);
