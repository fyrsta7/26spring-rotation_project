static int disassemble(RAsm *a, RAsmOp *op, const ut8 *buf, int len) {
	int opsize;
	static ud_t d;
	ud_init (&d);
	ud_set_syntax (&d, (a->syntax==R_ASM_SYNTAX_ATT)?
			UD_SYN_ATT: UD_SYN_INTEL);
	ud_set_input_buffer (&d, (uint8_t*) buf, len);
	ud_set_pc (&d, a->pc);
	ud_set_mode (&d, a->bits);
	opsize = ud_disassemble (&d);
	snprintf (op->buf_asm, R_ASM_BUFSIZE, "%s", ud_insn_asm (&d));
	op->size = opsize;
	if (opsize<1 || strstr (op->buf_asm, "invalid"))
		opsize = -1;
	return opsize;
}
