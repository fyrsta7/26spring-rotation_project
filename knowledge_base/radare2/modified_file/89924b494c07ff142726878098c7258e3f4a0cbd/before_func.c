static bool internal_esil_reg_write(REsil *esil, const char *regname, ut64 num) {
	r_return_val_if_fail (esil && esil->anal, false);
	RRegItem *ri = r_reg_get (esil->anal->reg, regname, -1);
	if (ri) {
		r_reg_set_value (esil->anal->reg, ri, num);
		R_LOG_DEBUG ("%s = %x", regname, (int)num);
		r_unref (ri);
		return true;
	}
	return false;
}
