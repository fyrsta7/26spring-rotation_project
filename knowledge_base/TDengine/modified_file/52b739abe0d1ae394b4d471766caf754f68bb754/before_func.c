        ++j;
        continue;
      }

      int32_t nextPos = (++i);
      if (nextPos != j) {
        memmove(pBlockData + dataBuf->rowSize * nextPos, pBlockData + dataBuf->rowSize * j, dataBuf->rowSize);
      }

      ++j;
    }

    dataBuf->ordered = true;

    pBlocks->numOfRows = i + 1;
    dataBuf->size = sizeof(SSubmitBlk) + dataBuf->rowSize * pBlocks->numOfRows;
  }
}

static int32_t doParseInsertStatement(SSqlCmd* pCmd, char **str, STableDataBlocks* dataBuf, int32_t *totalNum) {
  STableComInfo tinfo = tscGetTableInfo(dataBuf->pTableMeta);
  
  int32_t maxNumOfRows;
  int32_t code = tscAllocateMemIfNeed(dataBuf, tinfo.rowSize, &maxNumOfRows);
  if (TSDB_CODE_SUCCESS != code) {
    return TSDB_CODE_TSC_OUT_OF_MEMORY;
  }

  code = TSDB_CODE_TSC_INVALID_SQL;
  char  *tmpTokenBuf = calloc(1, 16*1024);  // used for deleting Escape character: \\, \', \"
  if (NULL == tmpTokenBuf) {
    return TSDB_CODE_TSC_OUT_OF_MEMORY;
  }

  int32_t numOfRows = 0;
  code = tsParseValues(str, dataBuf, maxNumOfRows, pCmd, &numOfRows, tmpTokenBuf);

  free(tmpTokenBuf);
  if (code != TSDB_CODE_SUCCESS) {
    return code;
  }

