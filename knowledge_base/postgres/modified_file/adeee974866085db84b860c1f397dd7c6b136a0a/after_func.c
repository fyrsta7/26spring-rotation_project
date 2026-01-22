static void tqueueWalk(TQueueDestReceiver * tqueue, RemapClass walktype,
		   Datum value);
static void tqueueWalkRecord(TQueueDestReceiver * tqueue, Datum value);
static void tqueueWalkArray(TQueueDestReceiver * tqueue, Datum value);
static void tqueueWalkRange(TQueueDestReceiver * tqueue, Datum value);
static void tqueueSendTypmodInfo(TQueueDestReceiver * tqueue, int typmod,
					 TupleDesc tupledesc);
static void TupleQueueHandleControlMessage(TupleQueueReader *reader,
							   Size nbytes, char *data);
static HeapTuple TupleQueueHandleDataMessage(TupleQueueReader *reader,
							Size nbytes, HeapTupleHeader data);
static HeapTuple TupleQueueRemapTuple(TupleQueueReader *reader,
					 TupleDesc tupledesc, RemapInfo * remapinfo,
					 HeapTuple tuple);
static Datum TupleQueueRemap(TupleQueueReader *reader, RemapClass remapclass,
				Datum value);
static Datum TupleQueueRemapArray(TupleQueueReader *reader, Datum value);
static Datum TupleQueueRemapRange(TupleQueueReader *reader, Datum value);
static Datum TupleQueueRemapRecord(TupleQueueReader *reader, Datum value);
static RemapClass GetRemapClass(Oid typeid);
static RemapInfo *BuildRemapInfo(TupleDesc tupledesc);

/*
 * Receive a tuple.
 *
 * This is, at core, pretty simple: just send the tuple to the designated
 * shm_mq.  The complicated part is that if the tuple contains transient
 * record types (see lookup_rowtype_tupdesc), we need to send control
 * information to the shm_mq receiver so that those typemods can be correctly
 * interpreted, as they are merely held in a backend-local cache.  Worse, the
 * record type may not at the top level: we could have a range over an array
 * type over a range type over a range type over an array type over a record,
 * or something like that.
 */
static void
tqueueReceiveSlot(TupleTableSlot *slot, DestReceiver *self)
{
	TQueueDestReceiver *tqueue = (TQueueDestReceiver *) self;
	TupleDesc	tupledesc = slot->tts_tupleDescriptor;
	HeapTuple	tuple;

	/*
	 * Test to see whether the tupledesc has changed; if so, set up for the
	 * new tupledesc.  This is a strange test both because the executor really
	 * shouldn't change the tupledesc, and also because it would be unsafe if
	 * the old tupledesc could be freed and a new one allocated at the same
	 * address.  But since some very old code in printtup.c uses a similar
	 * test, we adopt it here as well.
	 */
	if (tqueue->tupledesc != tupledesc)
	{
		if (tqueue->remapinfo != NULL)
			pfree(tqueue->remapinfo);
		tqueue->remapinfo = BuildRemapInfo(tupledesc);
		tqueue->tupledesc = tupledesc;
	}

	tuple = ExecMaterializeSlot(slot);

	/*
	 * When, because of the types being transmitted, no record typemod mapping
	 * can be needed, we can skip a good deal of work.
	 */
	if (tqueue->remapinfo != NULL)
	{
		RemapInfo  *remapinfo = tqueue->remapinfo;
		AttrNumber	i;
		MemoryContext oldcontext = NULL;

		/* Deform the tuple so we can examine it, if not done already. */
		slot_getallattrs(slot);

