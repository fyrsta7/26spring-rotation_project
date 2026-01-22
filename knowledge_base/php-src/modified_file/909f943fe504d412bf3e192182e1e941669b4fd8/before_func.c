			}
		} else {
			ZEND_ASSERT(!(HT_FLAGS(ht) & HASH_FLAG_PACKED));
			if (nSize > ht->nTableSize) {
				void *new_data, *old_data = HT_GET_DATA_ADDR(ht);
				Bucket *old_buckets = ht->arData;
				nSize = zend_hash_check_size(nSize);
				ht->nTableSize = nSize;
				new_data = pemalloc(HT_SIZE_EX(nSize, HT_SIZE_TO_MASK(nSize)), GC_FLAGS(ht) & IS_ARRAY_PERSISTENT);
				ht->nTableMask = HT_SIZE_TO_MASK(ht->nTableSize);
				HT_SET_DATA_ADDR(ht, new_data);
				memcpy(ht->arData, old_buckets, sizeof(Bucket) * ht->nNumUsed);
				pefree(old_data, GC_FLAGS(ht) & IS_ARRAY_PERSISTENT);
				zend_hash_rehash(ht);
			}
		}
