  taosUnlockList(pSet->lockedBy+hash);

  taosDecRsetCount(pSet);

  return p;
}

int taosReleaseRef(int rsetId, int64_t rid)
{
  return taosDecRefCount(rsetId, rid, 0);
}

// if rid is 0, return the first p in hash list, otherwise, return the next after current rid
void *taosIterateRef(int rsetId, int64_t rid) {
  SRefNode *pNode = NULL;
  SRefSet  *pSet;

  if (rsetId < 0 || rsetId >= TSDB_REF_OBJECTS) {
    uTrace("rsetId:%d rid:%" PRId64 " failed to iterate, rsetId not valid", rsetId, rid);
    terrno = TSDB_CODE_REF_INVALID_ID;
    return NULL;
  }

  if (rid < 0) {
    uTrace("rsetId:%d rid:%" PRId64 " failed to iterate, rid not valid", rsetId, rid);
    terrno = TSDB_CODE_REF_NOT_EXIST;
    return NULL;
  }

  void *newP = NULL;
  pSet = tsRefSetList + rsetId;
  taosIncRsetCount(pSet);
  if (pSet->state != TSDB_REF_STATE_ACTIVE) {
    uTrace("rsetId:%d rid:%" PRId64 " failed to iterate, rset not active", rsetId, rid);
    terrno = TSDB_CODE_REF_ID_REMOVED;
    taosDecRsetCount(pSet);
    return NULL;
  }

  do {
    newP = NULL;
    int hash = 0;
    if (rid > 0) {
      hash = rid % pSet->max;
      taosLockList(pSet->lockedBy+hash);

      pNode = pSet->nodeList[hash];
      while (pNode) {
        if (pNode->rid == rid) break;
        pNode = pNode->next;
      }

      if (pNode == NULL) {
        uError("rsetId:%d rid:%" PRId64 " not there, quit", rsetId, rid);
        terrno = TSDB_CODE_REF_NOT_EXIST;
        taosUnlockList(pSet->lockedBy+hash);
        taosDecRsetCount(pSet);
        return NULL;
      }

      // rid is there
      pNode = pNode->next;
      // check first place
      while (pNode) {
        if (!pNode->removed) break;
        pNode = pNode->next;
      }
      if (pNode == NULL) {
        taosUnlockList(pSet->lockedBy+hash);
        hash++;
      }
    }

    if (pNode == NULL) {
      for (; hash < pSet->max; ++hash) {
        taosLockList(pSet->lockedBy+hash);
        pNode = pSet->nodeList[hash];
        if (pNode) {
          // check first place
          while (pNode) {
            if (!pNode->removed) break;
            pNode = pNode->next;
          }
          if (pNode) break;
        }
        taosUnlockList(pSet->lockedBy+hash);
      }
    }

    if (pNode) {
      pNode->count++;  // acquire it
      newP = pNode->p;
      taosUnlockList(pSet->lockedBy+hash);
