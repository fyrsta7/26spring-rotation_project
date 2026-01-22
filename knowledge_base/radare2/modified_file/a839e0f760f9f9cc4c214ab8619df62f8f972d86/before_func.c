}

static int bin_dwarf(RCore *core, PJ *pj, int mode) {
	RBinDwarfRow *row;
	RListIter *iter;
	if (!r_config_get_i (core->config, "bin.dbginfo")) {
		return false;
	}
	RBinFile *binfile = r_bin_cur (core->bin);
	RBinPlugin * plugin = r_bin_file_cur_plugin (binfile);
	if (!binfile) {
		return false;
	}
	RList *list = NULL;
	RList *ownlist = NULL;
	if (plugin && plugin->lines) {
		// list is not cloned to improve speed. avoid use after free
		list = plugin->lines (binfile);
	} else if (core->bin) {
		// TODO: complete and speed-up support for dwarf
		RBinDwarfDebugAbbrev *da = NULL;
		da = r_bin_dwarf_parse_abbrev (core->bin, mode);
		RBinDwarfDebugInfo *info = r_bin_dwarf_parse_info (da, core->bin, mode);
		HtUP /*<offset, List *<LocListEntry>*/ *loc_table = r_bin_dwarf_parse_loc (core->bin, core->anal->bits / 8);
		// I suppose there is no reason the parse it for a printing purposes
		if (info && mode != R_MODE_PRINT) {
			/* Should we do this by default? */
			RAnalDwarfContext ctx = {
				.info = info,
				.loc = loc_table
			};
			r_anal_dwarf_process_info (core->anal, &ctx);
		}
		if (loc_table) {
			if (mode == R_MODE_PRINT) {
				r_bin_dwarf_print_loc (loc_table, core->anal->bits / 8, r_cons_printf);
			}
			r_bin_dwarf_free_loc (loc_table);
		}
		r_bin_dwarf_free_debug_info (info);
		r_bin_dwarf_parse_aranges (core->bin, mode);
		list = ownlist = r_bin_dwarf_parse_line (core->bin, mode);
		r_bin_dwarf_free_debug_abbrev (da);
	}
	if (!list) {
		return false;
	}

	r_cons_break_push (NULL, NULL);
	/* cache file:line contents */
	HtPP* file_lines = ht_pp_new (NULL, file_lines_free_kv, NULL);

	if (IS_MODE_JSON (mode)) {
		pj_a (pj);
	}

	//TODO we should need to store all this in sdb, or do a filecontentscache in libr/util
	//XXX this whole thing has leaks
	r_list_foreach (list, iter, row) {
		if (r_cons_is_breaked ()) {
			break;
		}
		if (mode) {
			// TODO: use 'Cl' instead of CC
			const char *path = row->file;
			FileLines *current_lines = ht_pp_find (file_lines, path, NULL);
			if (!current_lines) {
				current_lines = read_file_lines (path);
				if (!ht_pp_insert (file_lines, path, current_lines)) {
					file_lines_free (current_lines);
					current_lines = NULL;
				}
			}
			char *line = NULL;

			if (current_lines) {
				int nl = row->line - 1;
				if (nl >= 0 && nl < current_lines->line_count) {
					line = strdup (current_lines->content + current_lines->line_starts[nl]);
				}
			}
			if (line) {
				r_str_filter (line, strlen (line));
				line = r_str_replace (line, "\"", "\\\"", 1);
				line = r_str_replace (line, "\\\\", "\\", 1);
			}
			bool chopPath = !r_config_get_i (core->config, "dir.dwarf.abspath");
			char *file = strdup (row->file);
			if (chopPath) {
				const char *slash = r_str_lchr (file, '/');
				if (slash) {
					memmove (file, slash + 1, strlen (slash));
				}
			}
			// TODO: implement internal : if ((mode & R_MODE_SET))
			if ((mode & R_MODE_SET)) {
				// TODO: use CL here.. but its not necessary.. so better not do anything imho
				// r_core_cmdf (core, "CL %s:%d 0x%08"PFMT64x, file, (int)row->line, row->address);
#if 0
				char *cmt = r_str_newf ("%s:%d %s", file, (int)row->line, r_str_get (line));
				r_meta_set_string (core->anal, R_META_TYPE_COMMENT, row->address, cmt);
				free (cmt);
#endif
			} else if (IS_MODE_JSON(mode)) {
				pj_a (pj);

				pj_o (pj);
				pj_ks (pj, "name", "CC");
				pj_ks (pj, "file", file);
				pj_ki (pj, "line_num", (int) row->line);
				pj_kn (pj, "addr", row->address);
				pj_end (pj);

				pj_o (pj);
				pj_ks (pj, "name", "CL");
				pj_ks (pj, "file", file);
				pj_ki (pj, "line_num", (int) row->line);
				pj_ks (pj, "line", r_str_get (line));
				pj_kn (pj, "addr", row->address);
				pj_end (pj);

				pj_end (pj);
			} else {
				r_cons_printf ("CL %s:%d 0x%08" PFMT64x "\n",
					       file, (int)row->line,
					       row->address);
				r_cons_printf ("\"CC %s:%d %s\"@0x%" PFMT64x
					       "\n",
					       file, row->line,
					       r_str_get (line), row->address);
			}
			free (file);
			free (line);
		} else {
			r_cons_printf ("0x%08" PFMT64x "\t%s\t%d\n",
				       row->address, row->file, row->line);
		}
	}
	if (IS_MODE_JSON(mode)) {
		pj_end (pj);
	}
	r_cons_break_pop ();
	ht_pp_free (file_lines);
	r_list_free (ownlist);
