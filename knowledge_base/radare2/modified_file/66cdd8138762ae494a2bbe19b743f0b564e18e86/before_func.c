}

R_API void r_core_visual_title(RCore *core, int color) {
	bool showDelta = r_config_get_i (core->config, "scr.slow");
	static ut64 oldpc = 0;
	const char *BEGIN = core->cons->pal.prompt;
	const char *filename;
	char pos[512], bar[512], pcs[32];
	if (!oldpc) {
		oldpc = r_debug_reg_get (core->dbg, "PC");
	}
	/* automatic block size */
	int pc, hexcols = r_config_get_i (core->config, "hex.cols");
	if (autoblocksize) {
		switch (core->printidx) {
		case R_CORE_VISUAL_MODE_PXR: // prc
		case R_CORE_VISUAL_MODE_PRC: // prc
			r_core_block_size (core, (int)(core->cons->rows * hexcols * 3.5));
			break;
		case R_CORE_VISUAL_MODE_PX: // x
		case R_CORE_VISUAL_MODE_PXa: // pxa
			r_core_block_size (core, (int)(core->cons->rows * hexcols * 3.5));
			break;
		case R_CORE_VISUAL_MODE_PW: // XXX pw
			r_core_block_size (core, (int)(core->cons->rows * hexcols));
			break;
		case R_CORE_VISUAL_MODE_PC: // XXX pc
			r_core_block_size (core, (int)(core->cons->rows * hexcols * 4));
			break;
		case R_CORE_VISUAL_MODE_PD: // pd
		case R_CORE_VISUAL_MODE_PDDBG: // pd+dbg
		{
			int bsize = core->cons->rows * 5;

			if (core->print->screen_bounds > 1) {
				// estimate new blocksize with the size of the last
				// printed instructions
				int new_sz = core->print->screen_bounds - core->offset + 32;
				if (new_sz > bsize) {
					bsize = new_sz;
				}
			}
			r_core_block_size (core, bsize);
			break;
		}
		case R_CORE_VISUAL_MODE_PXA: // pxA
			r_core_block_size (core, hexcols * core->cons->rows * 8);
			break;
		}
	}
	if (r_config_get_i (core->config, "scr.zoneflags")) {
		r_core_cmd (core, "fz:", 0);
	}
	if (r_config_get_i (core->config, "cfg.debug")) {
		ut64 curpc = r_debug_reg_get (core->dbg, "PC");
		if (curpc && curpc != UT64_MAX && curpc != oldpc) {
			// check dbg.follow here
			int follow = (int) (st64) r_config_get_i (core->config, "dbg.follow");
			if (follow > 0) {
				if ((curpc < core->offset) || (curpc > (core->offset + follow))) {
					r_core_seek (core, curpc, 1);
				}
			} else if (follow < 0) {
				r_core_seek (core, curpc + follow, 1);
			}
			oldpc = curpc;
		}
	}

	RIODesc *desc = core->file? r_io_desc_get (core->io, core->file->fd): NULL;
	filename = desc? desc->name: "";
	{ /* get flag with delta */
		ut64 addr = core->offset + (core->print->cur_enabled? core->print->cur: 0);
#if 1
		/* TODO: we need a helper into r_flags to do that */
		bool oss = core->flags->space_strict;
		int osi = core->flags->space_idx;
		RFlagItem *f = NULL;
		core->flags->space_strict = true;
		core->anal->flb.set_fs (core->flags, "symbols");
		if (core->flags->space_idx != -1) {
			f = core->anal->flb.get_at (core->flags, addr, showDelta);
		}
		core->flags->space_strict = oss;
		core->flags->space_idx = osi;
		if (!f) {
			f = r_flag_get_at (core->flags, addr, showDelta);
		}
#else
		RFlagItem *f = r_flag_get_at (core->flags, addr, false);
#endif
		if (f) {
			if (f->offset == addr || !f->offset) {
				snprintf (pos, sizeof (pos), "@ %s", f->name);
			} else {
				snprintf (pos, sizeof (pos), "@ %s+%d # 0x%"PFMT64x,
					f->name, (int) (addr - f->offset), addr);
			}
		} else {
			RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, addr, 0);
			if (fcn) {
				int delta = addr - fcn->addr;
				if (delta > 0) {
					snprintf (pos, sizeof (pos), "@ %s+%d", fcn->name, delta);
				} else if (delta < 0) {
					snprintf (pos, sizeof (pos), "@ %s%d", fcn->name, delta);
				} else {
					snprintf (pos, sizeof (pos), "@ %s", fcn->name);
				}
			} else {
				pos[0] = 0;
			}
		}
	}

	if (core->print->cur < 0) {
		core->print->cur = 0;
	}

	if (color) {
		r_cons_strcat (BEGIN);
	}
	const char *cmd_visual = r_config_get (core->config, "cmd.visual");
	if (cmd_visual && *cmd_visual) {
		strncpy (bar, cmd_visual, sizeof (bar) - 1);
		bar[10] = '.'; // chop cmdfmt
		bar[11] = '.'; // chop cmdfmt
		bar[12] = 0; // chop cmdfmt
	} else {
		strncpy (bar, printfmt[PIDX], sizeof (bar) - 1);
		bar[sizeof (bar) - 1] = 0; // '\0'-terminate bar
		bar[10] = '.'; // chop cmdfmt
		bar[11] = '.'; // chop cmdfmt
		bar[12] = 0; // chop cmdfmt
	}
	{
		ut64 sz = r_io_size (core->io);
		ut64 pa;
		{
			RIOSection *s = r_io_section_vget (core->io, core->offset);
			pa =  s ? core->offset - s->vaddr + s->paddr : core->offset;
		}
		if (sz == UT64_MAX) {
			pcs[0] = 0;
		} else {
			if (!sz || pa > sz) {
				pc = 0;
			} else {
				pc = (pa * 100) / sz;
			}
			sprintf (pcs, "%d%% ", pc);
		}
	}
	{
		char *title;
		if (__ime) {
			title = r_str_newf ("[0x%08"PFMT64x " + %d> * INSERT MODE *\n",
				core->offset, core->print->cur);
		} else {
			if (core->print->cur_enabled) {
				if (core->print->ocur == -1) {
					title = r_str_newf ("[0x%08"PFMT64x " *0x%08"PFMT64x" ($$+0x%x)]> %s %s\n",
						core->offset, core->offset + core->print->cur,
						core->print->cur,
						bar, pos);
				} else {
					title = r_str_newf ("[0x%08"PFMT64x " 0x%08"PFMT64x" [0x%x..0x%x] %d]> %s %s\n",
						core->offset, core->offset + core->print->cur,
						core->print->ocur, core->print->cur, R_ABS (core->print->cur - core->print->ocur) + 1,
						bar, pos);
				}
			} else {
				title = r_str_newf ("[0x%08"PFMT64x " %s%d %s]> %s %s\n",
					core->offset, pcs, core->blocksize, filename, bar, pos);
			}
		}
		const int tabsCount = core->visual.tabs? r_list_length (core->visual.tabs): 0;
		if (tabsCount > 0) {
			const int curTab = core->visual.tab;
			r_cons_printf ("[");
			int i;
			for (i = 0; i < tabsCount; i++) {
				if (i == curTab - 1) {
					r_cons_printf ("%d", curTab);
				} else {
					r_cons_printf (".");
				}
			}
			r_cons_printf ("]");
			// r_cons_printf ("[tab:%d/%d]", core->visual.tab, tabsCount);
		}
		r_cons_print (title);
		free (title);
	}
	if (color) {
		r_cons_strcat (Color_RESET);
