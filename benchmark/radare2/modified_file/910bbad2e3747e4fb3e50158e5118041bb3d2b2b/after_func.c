static int gbDisass(RAsmOp *op, const ut8 *buf, ut64 len){
	int foo = gbOpLength (gb_op[buf[0]].type);
	if (len<foo)
		return 0;
	switch (gb_op[buf[0]].type) {
	case GB_8BIT:
		sprintf (op->buf_asm, "%s", gb_op[buf[0]].name);
		break;
	case GB_16BIT:
		sprintf (op->buf_asm, "%s %s", cb_ops[buf[1]>>3], cb_regs[buf[1]&7]);
		break;
	case GB_8BIT+ARG_8:
		sprintf (op->buf_asm, gb_op[buf[0]].name, buf[1]);
		break;
	case GB_8BIT+ARG_16:
		sprintf (op->buf_asm, gb_op[buf[0]].name, buf[1]+0x100*buf[2]);
		break;
	case GB_8BIT+ARG_8+GB_IO:
		sprintf (op->buf_asm, gb_op[buf[0]].name, 0xff00+buf[1]);
		break;
	}
	return foo;
}