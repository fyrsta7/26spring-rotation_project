}

R_API void r_core_anal_cc_init(RCore *core) {
	const char *dir_prefix = r_config_get (core->config, "dir.prefix");
	sdb_reset (core->anal->sdb_cc);
	const char *anal_arch = r_config_get (core->config, "anal.arch");
	int bits = core->anal->bits;
	char *dbpath = sdb_fmt ("%s/"R2_SDB_FCNSIGN"/cc-%s-%d.sdb", dir_prefix, anal_arch, bits);
	if (r_file_exists (dbpath)) {
		sdb_concat_by_path (core->anal->sdb_cc, dbpath);
