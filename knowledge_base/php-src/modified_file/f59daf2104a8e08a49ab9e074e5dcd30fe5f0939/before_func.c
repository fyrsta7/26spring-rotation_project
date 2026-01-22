
static void zend_hash_do_resize(HashTable *ht);

#define CHECK_INIT(ht, packed) do {												\
	if (UNEXPECTED((ht)->nTableMask == 0)) {							\
		if (packed) { \
			(ht)->arData = (Bucket *) safe_pemalloc((ht)->nTableSize, sizeof(Bucket), 0, (ht)->u.flags & HASH_FLAG_PERSISTENT);	\
			(ht)->u.flags |= HASH_FLAG_PACKED; \
		} else { \
			(ht)->arData = (Bucket *) safe_pemalloc((ht)->nTableSize, sizeof(Bucket) + sizeof(uint32_t), 0, (ht)->u.flags & HASH_FLAG_PERSISTENT);	\
			(ht)->arHash = (uint32_t*)((ht)->arData + (ht)->nTableSize);	\
			memset((ht)->arHash, INVALID_IDX, (ht)->nTableSize * sizeof(uint32_t));	\
		} \
		(ht)->nTableMask = (ht)->nTableSize - 1;						\
	}																	\
} while (0)
 
static const uint32_t uninitialized_bucket = {INVALID_IDX};

ZEND_API void _zend_hash_init(HashTable *ht, uint32_t nSize, dtor_func_t pDestructor, zend_bool persistent ZEND_FILE_LINE_DC)
{
	uint32_t i = 3;

	SET_INCONSISTENT(HT_OK);

	if (nSize >= 0x80000000) {
		/* prevent overflow */
		ht->nTableSize = 0x80000000;
	} else {
		while ((1U << i) < nSize) {
