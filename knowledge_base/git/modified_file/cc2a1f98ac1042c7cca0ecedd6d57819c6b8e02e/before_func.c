
	show_ref(".have", oid);
}

static void write_head_info(void)
{
	static struct oidset seen = OIDSET_INIT;

	for_each_ref(show_ref_cb, &seen);
	for_each_alternate_ref(show_one_alternate_ref, &seen);
	oidset_clear(&seen);
	if (!sent_capabilities)
		show_ref("capabilities^{}", null_oid());

	advertise_shallow_grafts(1);
