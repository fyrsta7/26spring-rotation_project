#include "pq.h"

#include "reftable-record.h"
#include "system.h"
#include "basics.h"

int pq_less(struct pq_entry *a, struct pq_entry *b)
{
	struct strbuf ak = STRBUF_INIT;
	struct strbuf bk = STRBUF_INIT;
	int cmp = 0;
	reftable_record_key(&a->rec, &ak);
	reftable_record_key(&b->rec, &bk);

	cmp = strbuf_cmp(&ak, &bk);

	strbuf_release(&ak);
	strbuf_release(&bk);
