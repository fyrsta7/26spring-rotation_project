        SHashEntry *pNewEntry = pHashObj->hashList[j];
        pushfrontNodeInEntryList(pNewEntry, pNode);
      } else {
        break;
      }
    }

    if (pNode != NULL) {
      while ((pNext = pNode->next) != NULL) {
        int32_t j = HASH_INDEX(pNext->hashVal, pHashObj->capacity);
        if (j != i) {
          pe->num -= 1;

          pNode->next = pNext->next;
          pNext->next = NULL;

          // added into new slot
          SHashEntry *pNewEntry = pHashObj->hashList[j];
