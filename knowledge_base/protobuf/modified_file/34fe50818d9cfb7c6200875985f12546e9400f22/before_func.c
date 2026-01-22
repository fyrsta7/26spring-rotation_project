static struct upb_strtable_entry *strent(struct upb_strtable *t, int32_t i) {
  return UPB_INDEX(t->t.entries, i-1, t->t.entry_size);
}

void upb_table_init(struct upb_table *t, uint32_t size, uint16_t entry_size)
{
  t->count = 0;
  t->entry_size = entry_size;
  t->size_lg2 = 1;
  while(size >>= 1) t->size_lg2++;
  t->size_lg2 = max(t->size_lg2, 4);  /* Min size of 16. */
