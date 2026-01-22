static void cmd_anal_calls(RCore *core, const char *input) {
	int minop = 1; // 4
	ut8 buf[32];
	RBinFile *binfile;
	const char *searchin = r_config_get (core->config, "search.in");
	RAnalOp op;
	ut64 addr, addr_end;
	ut64 len = r_num_math (core->num, input);
	if (len > 0xffffff) {
		eprintf ("Too big\n");
		return;
	}
	binfile = r_core_bin_cur (core);
	if (!binfile) {
		eprintf ("cur binfile null\n");
		return;
	}
	if (!len || (searchin && !strcmp (searchin, "file"))) {
		len = binfile->size;
	} else {
		if (len<1) {
			len = r_num_math (core->num, "$SS-($$-$S)"); // section size
		}
		if (core->offset + len > binfile->size) {
			len = binfile->size - core->offset;
		}
	}
	len = binfile->size;
	addr = core->offset;
	addr_end = addr + len;
	r_cons_break (NULL, NULL);
	while (addr < addr_end) {
		if (core->cons->breaked)
			break;
		// TODO: too many ioreads here
		r_io_read_at (core->io, addr, buf, sizeof (buf));
		if (r_anal_op (core->anal, &op, addr, buf, sizeof (buf))) {
			if (op.size < 1) op.size = minop; // XXX must be +4 on arm/mips/.. like we do in disasm.c
			if (op.type == R_ANAL_OP_TYPE_CALL) {
				r_core_anal_fcn (core, op.jump, UT64_MAX,
						R_ANAL_REF_TYPE_NULL, 16);
			}
		} else {
			op.size = minop;
		}
		addr += (op.size>0)? op.size: 1;
	}
}
