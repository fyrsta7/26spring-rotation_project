	/* data follows */
} ReorderBufferDiskChange;

#define IsSpecInsert(action) \
( \
	((action) == REORDER_BUFFER_CHANGE_INTERNAL_SPEC_INSERT) \
)
#define IsSpecConfirmOrAbort(action) \
( \
	(((action) == REORDER_BUFFER_CHANGE_INTERNAL_SPEC_CONFIRM) || \
	((action) == REORDER_BUFFER_CHANGE_INTERNAL_SPEC_ABORT)) \
)
#define IsInsertOrUpdate(action) \
( \
	(((action) == REORDER_BUFFER_CHANGE_INSERT) || \
	((action) == REORDER_BUFFER_CHANGE_UPDATE) || \
	((action) == REORDER_BUFFER_CHANGE_INTERNAL_SPEC_INSERT)) \
)

/*
 * Maximum number of changes kept in memory, per transaction. After that,
 * changes are spooled to disk.
 *
 * The current value should be sufficient to decode the entire transaction
 * without hitting disk in OLTP workloads, while starting to spool to disk in
 * other workloads reasonably fast.
 *
 * At some point in the future it probably makes sense to have a more elaborate
 * resource management here, but it's not entirely clear what that would look
 * like.
 */
int			logical_decoding_work_mem;
static const Size max_changes_in_memory = 4096; /* XXX for restore only */

/* GUC variable */
int			debug_logical_replication_streaming = DEBUG_LOGICAL_REP_STREAMING_BUFFERED;

/* ---------------------------------------
 * primary reorderbuffer support routines
 * ---------------------------------------
 */
static ReorderBufferTXN *ReorderBufferGetTXN(ReorderBuffer *rb);
static void ReorderBufferReturnTXN(ReorderBuffer *rb, ReorderBufferTXN *txn);
static ReorderBufferTXN *ReorderBufferTXNByXid(ReorderBuffer *rb,
											   TransactionId xid, bool create, bool *is_new,
											   XLogRecPtr lsn, bool create_as_top);
static void ReorderBufferTransferSnapToParent(ReorderBufferTXN *txn,
											  ReorderBufferTXN *subtxn);

static void AssertTXNLsnOrder(ReorderBuffer *rb);

/* ---------------------------------------
 * support functions for lsn-order iterating over the ->changes of a
 * transaction and its subtransactions
 *
 * used for iteration over the k-way heap merge of a transaction and its
 * subtransactions
 * ---------------------------------------
 */
static void ReorderBufferIterTXNInit(ReorderBuffer *rb, ReorderBufferTXN *txn,
									 ReorderBufferIterTXNState *volatile *iter_state);
static ReorderBufferChange *ReorderBufferIterTXNNext(ReorderBuffer *rb, ReorderBufferIterTXNState *state);
static void ReorderBufferIterTXNFinish(ReorderBuffer *rb,
									   ReorderBufferIterTXNState *state);
static void ReorderBufferExecuteInvalidations(uint32 nmsgs, SharedInvalidationMessage *msgs);

/*
 * ---------------------------------------
 * Disk serialization support functions
 * ---------------------------------------
 */
static void ReorderBufferCheckMemoryLimit(ReorderBuffer *rb);
static void ReorderBufferSerializeTXN(ReorderBuffer *rb, ReorderBufferTXN *txn);
static void ReorderBufferSerializeChange(ReorderBuffer *rb, ReorderBufferTXN *txn,
										 int fd, ReorderBufferChange *change);
