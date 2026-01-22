					     unsigned flags)
{
	if (!r->objects->kept_pack_cache.packs)
		return;
	if (r->objects->kept_pack_cache.flags == flags)
		return;
	FREE_AND_NULL(r->objects->kept_pack_cache.packs);
	r->objects->kept_pack_cache.flags = 0;
}

static struct packed_git **kept_pack_cache(struct repository *r, unsigned flags)
{
	maybe_invalidate_kept_pack_cache(r, flags);

	if (!r->objects->kept_pack_cache.packs) {
		struct packed_git **packs = NULL;
