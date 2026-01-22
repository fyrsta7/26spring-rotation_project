		arg = next + 1;
	}
}

static RList *foreach3list(RCore *core, char type, const char *glob) {
	bool va = r_config_get_b (core->config, "io.va");
	RList *list = r_list_newf (foreach3list_free);
	RListIter *iter;
	int i;
	switch (type) {
	case 'C':
		{
			RIntervalTreeIter it;
			RAnalMetaItem *meta;
			r_interval_tree_foreach (&core->anal->meta, it, meta) {
				if (meta->type != R_META_TYPE_COMMENT) {
					continue;
				}
				if (!glob || (meta->str && r_str_glob (meta->str, glob))) {
					ut64 addr = r_interval_tree_iter_get (&it)->start;
					append_item (list, NULL, addr, UT64_MAX);
				}
			}
		}
		break;
	case 'm': // @@@m
		{
			int fd = r_io_fd_get_current (core->io);
			// only iterate maps of current fd
			RList *maps = r_io_map_get_by_fd (core->io, fd);
			RIOMap *map;
			if (maps) {
				RListIter *iter;
				r_list_foreach (maps, iter, map) {
					append_item (list, NULL, r_io_map_begin (map), r_io_map_size (map));
				}
				r_list_free (maps);
			}
		}
		break;
	case 'M': // @@@M
		if (core->dbg && core->dbg->h && core->dbg->maps) {
			RDebugMap *map;
			r_list_foreach (core->dbg->maps, iter, map) {
				append_item (list, NULL, map->addr, map->size);
			}
		}
		break;
	case 'e': // @@@e
		{
			RBinAddr *entry;
			RList *elist = r_bin_get_entries (core->bin);
			r_list_foreach (elist, iter, entry) {
				ut64 addr = va? entry->vaddr: entry->paddr;
				append_item (list, NULL, addr, UT64_MAX);
			}
			r_list_free (elist);
		}
		break;
	case 't': // @@@t
		// iterate over all threads
		if (core->dbg && core->dbg->h && core->dbg->h->threads) {
			RDebugPid *p;
			RList *thlist = core->dbg->h->threads (core->dbg, core->dbg->pid);
			r_list_foreach (thlist, iter, p) {
				append_item (list, NULL, (ut64)p->pid, UT64_MAX);
			}
			r_list_free (thlist);
		}
		break;
	case 'i': // @@@i
		{
			RBinImport *imp;
			const RList *implist = r_bin_get_imports (core->bin);
			r_list_foreach (implist, iter, imp) {
				char *impflag = r_str_newf ("sym.imp.%s", imp->name);
				ut64 addr = r_num_math (core->num, impflag);
				if (addr != 0 && addr != UT64_MAX) {
					append_item (list, NULL, addr, UT64_MAX);
				}
				free (impflag);
			}
		}
		break;
	case 'E':
		{
			RBinSymbol *sym;
			RList *symlist = r_bin_get_symbols (core->bin);
			bool va = r_config_get_b (core->config, "io.va");
			r_list_foreach (symlist, iter, sym) {
				if (!isAnExport (sym)) {
					continue;
				}
				ut64 addr = va? sym->vaddr: sym->paddr;
				append_item (list, NULL, addr, UT64_MAX);
			}
		}
		break;
	case 's': // @@@s symbols
		{
			RBinSymbol *sym;
			RList *syms = r_bin_get_symbols (core->bin);
			r_list_foreach (syms, iter, sym) {
				ut64 addr = va? sym->vaddr: sym->paddr;
				append_item (list, NULL, addr, sym->size);
			}
		}
		break;
	case 'S': // "@@@S"
		{
			RBinObject *obj = r_bin_cur_object (core->bin);
			if (obj) {
				RBinSection *sec;
				r_list_foreach (obj->sections, iter, sec) {
					ut64 addr = va ? sec->vaddr: sec->paddr;
					ut64 size = va ? sec->vsize: sec->size;
					append_item (list, NULL, addr, size);
				}
			}
		}
		break;
	case 'z':
		{
			RList *zlist = r_bin_get_strings (core->bin);
			if (zlist) {
				RBinString *s;
				r_list_foreach (zlist, iter, s) {
					ut64 addr = va? s->vaddr: s->paddr;
					append_item (list, NULL, addr, s->size);
				}
			}
		}
		break;
	case 'b':
		{
			RAnalFunction *fcn = r_anal_get_fcn_in (core->anal, core->offset, 0);
			if (fcn) {
				RListIter *iter;
				RAnalBlock *bb;
				r_list_foreach (fcn->bbs, iter, bb) {
					append_item (list, NULL, bb->addr, bb->size);
				}
			}
		}
		break;
	case 'F':
		{
			RAnalFunction *fcn;
			r_list_foreach (core->anal->fcns, iter, fcn) {
				if (!glob || r_str_glob (fcn->name, glob)) {
					ut64 size = r_anal_function_linear_size (fcn);
					append_item (list, NULL, fcn->addr, size);
				}
			}
		}
		break;
	case 'R': // relocs
		{
			RRBTree *rels = r_bin_get_relocs (core->bin);
			if (rels) {
				RRBNode *node = r_crbtree_first_node (rels);
				while (node) {
					RBinReloc *rel = (RBinReloc *)node->data;
					ut64 addr = va? rel->vaddr: rel->paddr;
					append_item (list, NULL, addr, UT64_MAX);
					node = r_rbnode_next (node);
				}
			}
		}
		break;
	case 'r': // registers
		{
			const int bits = core->anal->config->bits;
			for (i = 0; i < R_REG_TYPE_LAST; i++) {
				RRegItem *item;
				RList *head = r_reg_get_list (core->dbg->reg, i);
				r_list_foreach (head, iter, item) {
					if (item->size != bits) {
						continue;
					}
					if (item->type != i) {
						continue;
					}
					ut64 addr = r_reg_getv (core->dbg->reg, item->name);
					append_item (list, item->name, addr, item->size);
				}
			}
		}
		break;
	case 'f':
		r_flag_foreach_glob (core->flags, glob, copy_into_flagitem_list, list);
