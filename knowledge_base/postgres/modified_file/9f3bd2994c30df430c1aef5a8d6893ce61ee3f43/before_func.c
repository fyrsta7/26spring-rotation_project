	snap.xmin = xmin;
	snap.xmax = xmax;
	snap.nxip = 0;

	buf = makeStringInfo();
	appendBinaryStringInfo(buf, (char *)&snap, TXID_SNAPSHOT_SIZE(0));
	return buf;
}

static void
buf_add_txid(StringInfo buf, txid xid)
{
	TxidSnapshot *snap = (TxidSnapshot *)buf->data;

	/* do it before possible realloc */
	snap->nxip++;

	appendBinaryStringInfo(buf, (char *)&xid, sizeof(xid));
}

static TxidSnapshot *
buf_finalize(StringInfo buf)
{
	TxidSnapshot *snap = (TxidSnapshot *)buf->data;
	SET_VARSIZE(snap, buf->len);
