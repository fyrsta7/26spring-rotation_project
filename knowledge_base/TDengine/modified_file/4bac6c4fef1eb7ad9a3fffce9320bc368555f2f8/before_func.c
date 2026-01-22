        pCtx[i].fpSet.init(&pCtx[i], pResInfo);
      } else {
        pResInfo->initialized = true;
      }
    }
  }
}

static void extractQualifiedTupleByFilterResult(SSDataBlock* pBlock, const int8_t* rowRes, bool keep);
void doFilter(const SNode* pFilterNode, SSDataBlock* pBlock, SArray* pColMatchInfo) {
  if (pFilterNode == NULL) {
    return;
  }

  SFilterInfo* filter = NULL;

  // todo move to the initialization function
  int32_t code = filterInitFromNode((SNode*)pFilterNode, &filter, 0);

  SFilterColumnParam param1 = {.numOfCols = pBlock->info.numOfCols, .pDataBlock = pBlock->pDataBlock};
  code = filterSetDataFromSlotId(filter, &param1);

  int8_t* rowRes = NULL;

  // todo the keep seems never to be True??
  bool keep = filterExecute(filter, pBlock, &rowRes, NULL, param1.numOfCols);
  filterFreeInfo(filter);

  extractQualifiedTupleByFilterResult(pBlock, rowRes, keep);
  blockDataUpdateTsWindow(pBlock);
}

void extractQualifiedTupleByFilterResult(SSDataBlock* pBlock, const int8_t* rowRes, bool keep) {
  if (keep) {
    return;
  }

  if (rowRes != NULL) {
    SSDataBlock* px = createOneDataBlock(pBlock, false);
    blockDataEnsureCapacity(px, pBlock->info.rows);

    int32_t totalRows = pBlock->info.rows;
