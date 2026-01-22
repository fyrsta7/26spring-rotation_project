		const char *ms = strstr (mn, "method.");
		if (ms) {
			mn = ms + strlen ("method.");
		}
		r_cons_printf ("  public %s ();\n", mn);
	}
	r_cons_printf ("}\n\n");
}

static int bin_classes(RCore *r, PJ *pj, int mode) {
	RListIter *iter, *iter2, *iter3;
	RBinSymbol *sym;
	RBinClass *c;
	RBinField *f;
	char *name;
	RList *cs = r_bin_get_classes (r->bin);
	if (!cs) {
		if (IS_MODE_JSON (mode)) {
			pj_a (pj);
			pj_end (pj);
			return true;
		}
		return false;
	}
	// XXX: support for classes is broken and needs more love
	if (IS_MODE_JSON (mode)) {
		pj_a (pj);
	} else if (IS_MODE_SET (mode)) {
		if (!r_config_get_b (r->config, "bin.classes")) {
			return false;
		}
		r_flag_space_set (r->flags, R_FLAGS_FS_CLASSES);
	} else if (IS_MODE_RAD (mode) && !IS_MODE_CLASSDUMP (mode)) {
		r_cons_println ("fs classes");
	}

	r_list_foreach (cs, iter, c) {
		if (!c || !c->name || !c->name[0]) {
			continue;
		}
		name = strdup (c->name);
		r_name_filter (name, -1);
		ut64 at_min = UT64_MAX;
		ut64 at_max = 0LL;

		r_list_foreach (c->methods, iter2, sym) {
			if (sym->vaddr) {
				if (sym->vaddr < at_min) {
					at_min = sym->vaddr;
				}
				if (sym->vaddr + sym->size > at_max) {
					at_max = sym->vaddr + sym->size;
				}
			}
		}
		if (at_min == UT64_MAX) {
			at_min = c->addr;
			at_max = c->addr; // XXX + size?
		}

		if (IS_MODE_SET (mode)) {
			r_strf_var (classname, 256, "class.%s", name);
			r_flag_set (r->flags, classname, c->addr, 1);
			r_list_foreach (c->methods, iter2, sym) {
				char *mflags = r_core_bin_method_flags_str (sym->method_flags, mode);
				r_strf_var (method, 256, "method%s.%s.%s", mflags, c->name, sym->name);
				R_FREE (mflags);
				r_name_filter (method, -1);
				r_flag_set (r->flags, method, sym->vaddr, 1);
			}
#if 1
			r_list_foreach (c->fields, iter2, f) {
				char *fn = r_str_newf ("field.%s.%s", classname, f->name);
				ut64 at = f->vaddr; //  sym->vaddr + (f->vaddr &  0xffff);
				r_flag_set (r->flags, fn, at, 1);
				free (fn);
			}
#endif
		} else if (IS_MODE_SIMPLEST (mode)) {
			r_cons_printf ("%s\n", c->name);
		} else if (IS_MODE_SIMPLE (mode)) {
			r_cons_printf ("0x%08"PFMT64x" [0x%08"PFMT64x" - 0x%08"PFMT64x"] %s %s%s%s\n",
				c->addr, at_min, at_max, r_bin_lang_tostring (c->lang), c->name, c->super ? " " : "",
				r_str_get (c->super));
		} else if (IS_MODE_CLASSDUMP (mode)) {
			if (c) {
				RBinFile *bf = r_bin_cur (r->bin);
				if (bf && bf->o) {
					if (IS_MODE_RAD (mode)) {
						classdump_c (r, c);
					} else if (bf->o->lang == R_BIN_LANG_JAVA || (bf->o->info && bf->o->info->lang && strstr (bf->o->info->lang, "dalvik"))) {
						classdump_java (r, c);
					} else {
						classdump_objc (r, c);
					}
				} else {
					classdump_objc (r, c);
				}
			}
		} else if (IS_MODE_RAD (mode)) {
			char *n = __filterShell (name);
			r_cons_printf ("\"f class.%s = 0x%"PFMT64x"\"\n", n, at_min);
			free (n);
			if (c->super) {
				char *cn = c->name; // __filterShell (c->name);
				char *su = c->super; // __filterShell (c->super);
				r_cons_printf ("\"f super.%s.%s = %d\"\n",
						cn, su, c->index);
				// free (cn);
				// free (su);
			}
			r_list_foreach (c->methods, iter2, sym) {
				char *mflags = r_core_bin_method_flags_str (sym->method_flags, mode);
				char *n = c->name; //  __filterShell (c->name);
				char *sn = sym->name; //__filterShell (sym->name);
				char *cmd = r_str_newf ("\"f method%s.%s.%s = 0x%"PFMT64x"\"\n", mflags, n, sn, sym->vaddr);
				// free (n);
				// free (sn);
				if (cmd) {
					r_str_replace_char (cmd, ' ', '_');
					if (strlen (cmd) > 2) {
						cmd[2] = ' ';
					}
					char *eq = (char *)r_str_rchr (cmd, NULL, '=');
					if (eq && eq != cmd) {
						eq[-1] = eq[1] = ' ';
					}
					r_str_replace_char (cmd, '\n', 0);
					r_cons_printf ("%s\n", cmd);
					free (cmd);
				}
				R_FREE (mflags);
			}
			r_list_foreach (c->fields, iter2, f) {
				char *fn = r_str_newf ("field.%s.%s", c->name, f->name);
				r_name_filter (fn, -1);
				ut64 at = f->vaddr; //  sym->vaddr + (f->vaddr &  0xffff);
				r_cons_printf ("\"f %s = 0x%08"PFMT64x"\"\n", fn, at);
				free (fn);
			}

			// C struct
			r_cons_printf ("\"td struct %s {", c->name);
			if (r_list_empty (c->fields)) {
				// XXX workaround because we cant register empty structs yet
				// XXX https://github.com/radareorg/radare2/issues/16342
				r_cons_printf (" char empty[0];");
			} else {
				r_list_foreach (c->fields, iter2, f) {
					char *n = objc_name_toc (f->name);
					char *t = objc_type_toc (f->type);
					r_cons_printf (" %s %s;", t, n);
					free (t);
					free (n);
				}
			}
			r_cons_printf ("};\"\n");
		} else if (IS_MODE_JSON (mode)) {
			pj_o (pj);
			pj_ks (pj, "classname", c->name);
			pj_kN (pj, "addr", c->addr);
			const char *lang = r_bin_lang_tostring (c->lang);
			if (lang && *lang != '?') {
				pj_ks (pj, "lang", lang);
			}
			pj_ki (pj, "index", c->index);
			if (c->super) {
				pj_ks (pj, "visibility", r_str_get (c->visibility_str));
				pj_ks (pj, "super", c->super);
			}
			pj_ka (pj, "methods");
			r_list_foreach (c->methods, iter2, sym) {
				pj_o (pj);
				pj_ks (pj, "name", sym->name);
				RFlagItem *fi = r_flag_get_at (r->flags, sym->vaddr, false);
				if (fi) {
					pj_ks (pj, "flag", fi->realname? fi->realname: fi->name);
				}
				char *s = r_core_cmd_strf (r, "isqq.@0x%08"PFMT64x"@e:bin.demangle=false", sym->vaddr);
				r_str_trim (s);
				if (R_STR_ISNOTEMPTY (s)) {
					pj_ks (pj, "realname", s);
				}
				free (s);
				if (sym->method_flags) {
					char *mflags = r_core_bin_method_flags_str (sym->method_flags, mode);
					pj_k (pj, "flags");
					pj_j (pj, mflags);
					free (mflags);
				}
				pj_kN (pj, "addr", sym->vaddr);
				pj_end (pj);
			}
			pj_end (pj);
			pj_ka (pj, "fields");
			r_list_foreach (c->fields, iter3, f) {
				pj_o (pj);
				pj_ks (pj, "name", f->name);
				if (f->flags) {
					char *mflags = r_core_bin_method_flags_str (f->flags, mode);
					pj_k (pj, "flags");
					pj_j (pj, mflags);
					free (mflags);
				}
				pj_kN (pj, "addr", f->vaddr);
				pj_end (pj);
			}
			pj_end (pj);
			pj_end (pj);
		} else {
			int m = 0;
			const char *cl = r_bin_lang_tostring (c->lang);
			r_cons_printf ("0x%08"PFMT64x" [0x%08"PFMT64x" - 0x%08"PFMT64x"] %6"PFMT64d" %s class %d %s",
				c->addr, at_min, at_max, (at_max - at_min), cl, c->index, c->name);
			if (c->super) {
				r_cons_printf (" :: %s\n", c->super);
			} else {
				r_cons_newline ();
			}
			r_list_foreach (c->methods, iter2, sym) {
				char *mflags = r_core_bin_method_flags_str (sym->method_flags, mode);
				const char *ls = r_bin_lang_tostring (sym->lang);
				r_cons_printf ("0x%08"PFMT64x" %s method %d %s %s\n",
					sym->vaddr, ls?ls:"?", m, mflags, sym->dname? sym->dname: sym->name);
				R_FREE (mflags);
				m++;
			}
			m = 0;
			const char *ls = r_bin_lang_tostring (c->lang);
			r_list_foreach (c->fields, iter3, f) {
				char *mflags = r_core_bin_method_flags_str (f->flags, mode);
				r_cons_printf ("0x%08"PFMT64x" %s field %d %s %s\n",
					f->vaddr, ls, m, mflags, f->name);
				m++;
			}
