static bool internal_esil_reg_write(REsil *esil, const char *regname, ut64 num) {
	R_RETURN_VAL_IF_FAIL (esil && esil->anal, false);
#if 1
	if (r_reg_setv (esil->anal->reg, regname, num)) {
		return true;
	}
	R_LOG_DEBUG ("Register %s does not exist", regname);
#else
	RRegItem *ri = r_reg_get (esil->anal->reg, regname, -1);
	if (ri) {
		r_reg_set_value (esil->anal->reg, ri, num);
		R_LOG_DEBUG ("%s = %x", regname, (int)num);
		r_unref (ri);
		return true;
	}
#endif
	return false;
}
