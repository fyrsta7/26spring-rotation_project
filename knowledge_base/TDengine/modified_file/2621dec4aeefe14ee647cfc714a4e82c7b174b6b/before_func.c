}

static STuplePos saveTupleData(SqlFunctionCtx* pCtx, int32_t rowIndex, const SSDataBlock* pSrcBlock,
                               const STupleKey* pKey);
static int32_t   updateTupleData(SqlFunctionCtx* pCtx, int32_t rowIndex, const SSDataBlock* pSrcBlock, STuplePos* pPos);
static const char* loadTupleData(SqlFunctionCtx* pCtx, const STuplePos* pPos);

static int32_t findRowIndex(int32_t start, int32_t num, SColumnInfoData* pCol, const char* tval) {
  // the data is loaded, not only the block SMA value
  for (int32_t i = start; i < num + start; ++i) {
    char* p = colDataGetData(pCol, i);
    if (memcmp((void*)tval, p, pCol->info.bytes) == 0) {
      return i;
    }
  }

  // if reach here means real data of block SMA is not set in pCtx->input.
  return -1;
}

int32_t doMinMaxHelper(SqlFunctionCtx* pCtx, int32_t isMinFunc) {
  int32_t numOfElems = 0;

  SInputColumnInfoData* pInput = &pCtx->input;
  SColumnDataAgg*       pAgg = pInput->pColumnDataAgg[0];

  SColumnInfoData* pCol = pInput->pData[0];
  int32_t          type = pCol->info.type;

  SResultRowEntryInfo* pResInfo = GET_RES_INFO(pCtx);
  SMinmaxResInfo*      pBuf = GET_ROWCELL_INTERBUF(pResInfo);
  pBuf->type = type;

  if (IS_NULL_TYPE(type)) {
    numOfElems = 0;
    goto _min_max_over;
  }

  // data in current data block are qualified to the query
  if (pInput->colDataAggIsSet) {
    numOfElems = pInput->numOfRows - pAgg->numOfNull;
    ASSERT(pInput->numOfRows == pInput->totalRows && numOfElems >= 0);
    if (numOfElems == 0) {
      return numOfElems;
    }

    void*   tval = NULL;
    int16_t index = 0;

    if (isMinFunc) {
      tval = &pInput->pColumnDataAgg[0]->min;
    } else {
      tval = &pInput->pColumnDataAgg[0]->max;
    }

    if (!pBuf->assign) {
      pBuf->v = *(int64_t*)tval;
      if (pCtx->subsidiaries.num > 0) {
        index = findRowIndex(pInput->startRowIndex, pInput->numOfRows, pCol, tval);
        if (index >= 0) {
          pBuf->tuplePos = saveTupleData(pCtx, index, pCtx->pSrcBlock, NULL);
        }
      }
    } else {
      if (IS_SIGNED_NUMERIC_TYPE(type)) {
        int64_t prev = 0;
        GET_TYPED_DATA(prev, int64_t, type, &pBuf->v);

        int64_t val = GET_INT64_VAL(tval);
        if ((prev < val) ^ isMinFunc) {
          *(int64_t*)&pBuf->v = val;
          if (pCtx->subsidiaries.num > 0) {
            index = findRowIndex(pInput->startRowIndex, pInput->numOfRows, pCol, tval);
            if (index >= 0) {
              pBuf->tuplePos = saveTupleData(pCtx, index, pCtx->pSrcBlock, NULL);
            }
          }
        }
      } else if (IS_UNSIGNED_NUMERIC_TYPE(type)) {
        uint64_t prev = 0;
        GET_TYPED_DATA(prev, uint64_t, type, &pBuf->v);

        uint64_t val = GET_UINT64_VAL(tval);
        if ((prev < val) ^ isMinFunc) {
          *(uint64_t*)&pBuf->v = val;
          if (pCtx->subsidiaries.num > 0) {
            index = findRowIndex(pInput->startRowIndex, pInput->numOfRows, pCol, tval);
            if (index >= 0) {
              pBuf->tuplePos = saveTupleData(pCtx, index, pCtx->pSrcBlock, NULL);
            }
          }
        }
      } else if (type == TSDB_DATA_TYPE_DOUBLE) {
        double prev = 0;
        GET_TYPED_DATA(prev, double, type, &pBuf->v);

        double val = GET_DOUBLE_VAL(tval);
        if ((prev < val) ^ isMinFunc) {
          *(double*)&pBuf->v = val;
          if (pCtx->subsidiaries.num > 0) {
            index = findRowIndex(pInput->startRowIndex, pInput->numOfRows, pCol, tval);
            if (index >= 0) {
              pBuf->tuplePos = saveTupleData(pCtx, index, pCtx->pSrcBlock, NULL);
            }
          }
        }
      } else if (type == TSDB_DATA_TYPE_FLOAT) {
        float prev = 0;
        GET_TYPED_DATA(prev, float, type, &pBuf->v);

        float val = GET_DOUBLE_VAL(tval);
        if ((prev < val) ^ isMinFunc) {
          *(float*)&pBuf->v = val;
        }

        if (pCtx->subsidiaries.num > 0) {
          index = findRowIndex(pInput->startRowIndex, pInput->numOfRows, pCol, tval);
          if (index >= 0) {
            pBuf->tuplePos = saveTupleData(pCtx, index, pCtx->pSrcBlock, NULL);
          }
        }
      }
    }

    pBuf->assign = true;
    return numOfElems;
  }

  int32_t start = pInput->startRowIndex;
  int32_t numOfRows = pInput->numOfRows;

  if (IS_SIGNED_NUMERIC_TYPE(type) || type == TSDB_DATA_TYPE_BOOL) {
    if (type == TSDB_DATA_TYPE_TINYINT || type == TSDB_DATA_TYPE_BOOL) {
      int8_t* pData = (int8_t*)pCol->pData;
      int8_t* val = (int8_t*)&pBuf->v;

      for (int32_t i = start; i < start + numOfRows; ++i) {
        if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
          continue;
        }

        if (!pBuf->assign) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
          }
          pBuf->assign = true;
        } else {
          // ignore the equivalent data value
          if ((*val) == pData[i]) {
            continue;
          }

          if ((*val < pData[i]) ^ isMinFunc) {
            *val = pData[i];
            if (pCtx->subsidiaries.num > 0) {
              updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
            }
          }
        }

        numOfElems += 1;
      }
    } else if (type == TSDB_DATA_TYPE_SMALLINT) {
      int16_t* pData = (int16_t*)pCol->pData;
      int16_t* val = (int16_t*)&pBuf->v;

      for (int32_t i = start; i < start + numOfRows; ++i) {
        if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
          continue;
        }

        if (!pBuf->assign) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
          }
          pBuf->assign = true;
        } else {
          // ignore the equivalent data value
          if ((*val) == pData[i]) {
            continue;
          }

          if ((*val < pData[i]) ^ isMinFunc) {
            *val = pData[i];
            if (pCtx->subsidiaries.num > 0) {
              updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
            }
          }
        }

        numOfElems += 1;
      }
    } else if (type == TSDB_DATA_TYPE_INT) {
      int32_t* pData = (int32_t*)pCol->pData;
      int32_t* val = (int32_t*)&pBuf->v;

      for (int32_t i = start; i < start + numOfRows; ++i) {
        if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
          continue;
        }

        if (!pBuf->assign) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
          }
          pBuf->assign = true;
        } else {
          // ignore the equivalent data value
          if ((*val) == pData[i]) {
            continue;
          }

          if ((*val < pData[i]) ^ isMinFunc) {
            *val = pData[i];
            if (pCtx->subsidiaries.num > 0) {
              updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
            }
          }
        }

        numOfElems += 1;
      }
    } else if (type == TSDB_DATA_TYPE_BIGINT) {
      int64_t* pData = (int64_t*)pCol->pData;
      int64_t* val = (int64_t*)&pBuf->v;

      for (int32_t i = start; i < start + numOfRows; ++i) {
        if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
          continue;
        }

        if (!pBuf->assign) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
          }
          pBuf->assign = true;
        } else {
          // ignore the equivalent data value
          if ((*val) == pData[i]) {
            continue;
          }

          if ((*val < pData[i]) ^ isMinFunc) {
            *val = pData[i];
            if (pCtx->subsidiaries.num > 0) {
              updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
            }
          }
        }

        numOfElems += 1;
      }
    }
  } else if (IS_UNSIGNED_NUMERIC_TYPE(type)) {
    if (type == TSDB_DATA_TYPE_UTINYINT) {
      uint8_t* pData = (uint8_t*)pCol->pData;
      uint8_t* val = (uint8_t*)&pBuf->v;

      for (int32_t i = start; i < start + numOfRows; ++i) {
        if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
          continue;
        }

        if (!pBuf->assign) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
          }
          pBuf->assign = true;
        } else {
          // ignore the equivalent data value
          if ((*val) == pData[i]) {
            continue;
          }

          if ((*val < pData[i]) ^ isMinFunc) {
            *val = pData[i];
            if (pCtx->subsidiaries.num > 0) {
              updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
            }
          }
        }

        numOfElems += 1;
      }
    } else if (type == TSDB_DATA_TYPE_USMALLINT) {
      uint16_t* pData = (uint16_t*)pCol->pData;
      uint16_t* val = (uint16_t*)&pBuf->v;

      for (int32_t i = start; i < start + numOfRows; ++i) {
        if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
          continue;
        }

        if (!pBuf->assign) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
          }
          pBuf->assign = true;
        } else {
          // ignore the equivalent data value
          if ((*val) == pData[i]) {
            continue;
          }

          if ((*val < pData[i]) ^ isMinFunc) {
            *val = pData[i];
            if (pCtx->subsidiaries.num > 0) {
              updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
            }
          }
        }

        numOfElems += 1;
      }
    } else if (type == TSDB_DATA_TYPE_UINT) {
      uint32_t* pData = (uint32_t*)pCol->pData;
      uint32_t* val = (uint32_t*)&pBuf->v;

      for (int32_t i = start; i < start + numOfRows; ++i) {
        if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
          continue;
        }

        if (!pBuf->assign) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
          }
          pBuf->assign = true;
        } else {
          // ignore the equivalent data value
          if ((*val) == pData[i]) {
            continue;
          }

          if ((*val < pData[i]) ^ isMinFunc) {
            *val = pData[i];
            if (pCtx->subsidiaries.num > 0) {
              updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
            }
          }
        }

        numOfElems += 1;
      }
    } else if (type == TSDB_DATA_TYPE_UBIGINT) {
      uint64_t* pData = (uint64_t*)pCol->pData;
      uint64_t* val = (uint64_t*)&pBuf->v;

      for (int32_t i = start; i < start + numOfRows; ++i) {
        if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
          continue;
        }

        if (!pBuf->assign) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
          }
          pBuf->assign = true;
        } else {
          // ignore the equivalent data value
          if ((*val) == pData[i]) {
            continue;
          }

          if ((*val < pData[i]) ^ isMinFunc) {
            *val = pData[i];
            if (pCtx->subsidiaries.num > 0) {
              updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
            }
          }
        }

        numOfElems += 1;
      }
    }
  } else if (type == TSDB_DATA_TYPE_DOUBLE) {
    double* pData = (double*)pCol->pData;
    double* val = (double*)&pBuf->v;

    for (int32_t i = start; i < start + numOfRows; ++i) {
      if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
        continue;
      }

      if (!pBuf->assign) {
        *val = pData[i];
        if (pCtx->subsidiaries.num > 0) {
          pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
        }
        pBuf->assign = true;
      } else {
        // ignore the equivalent data value
        if ((*val) == pData[i]) {
          continue;
        }

        if ((*val < pData[i]) ^ isMinFunc) {
          *val = pData[i];
          if (pCtx->subsidiaries.num > 0) {
            updateTupleData(pCtx, i, pCtx->pSrcBlock, &pBuf->tuplePos);
          }
        }
      }

      numOfElems += 1;
    }
  } else if (type == TSDB_DATA_TYPE_FLOAT) {
    float* pData = (float*)pCol->pData;
    float* val = (float*)&pBuf->v;

    for (int32_t i = start; i < start + numOfRows; ++i) {
      if ((pCol->hasNull) && colDataIsNull_f(pCol->nullbitmap, i)) {
        continue;
      }

      if (!pBuf->assign) {
        *val = pData[i];
        if (pCtx->subsidiaries.num > 0) {
          pBuf->tuplePos = saveTupleData(pCtx, i, pCtx->pSrcBlock, NULL);
        }
        pBuf->assign = true;
      } else {
        // ignore the equivalent data value
        if ((*val) == pData[i]) {
          continue;
        }
