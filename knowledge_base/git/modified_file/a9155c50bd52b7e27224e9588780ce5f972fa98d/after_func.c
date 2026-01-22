		      "If you are sure you want to delete it, "
		      "run 'git branch -D %s'."), branchname, branchname);
		return -1;
	}
	return 0;
}

static void delete_branch_config(const char *branchname)
{
	struct strbuf buf = STRBUF_INIT;
	strbuf_addf(&buf, "branch.%s", branchname);
	if (git_config_rename_section(buf.buf, NULL) < 0)
		warning(_("Update of config-file failed"));
	strbuf_release(&buf);
}

static int delete_branches(int argc, const char **argv, int force, int kinds,
			   int quiet)
{
	struct commit *head_rev = NULL;
	struct object_id oid;
	char *name = NULL;
	const char *fmt;
	int i;
	int ret = 0;
	int remote_branch = 0;
	struct strbuf bname = STRBUF_INIT;
	unsigned allowed_interpret;

	switch (kinds) {
	case FILTER_REFS_REMOTES:
		fmt = "refs/remotes/%s";
		/* For subsequent UI messages */
		remote_branch = 1;
		allowed_interpret = INTERPRET_BRANCH_REMOTE;

		force = 1;
		break;
	case FILTER_REFS_BRANCHES:
		fmt = "refs/heads/%s";
		allowed_interpret = INTERPRET_BRANCH_LOCAL;
		break;
	default:
		die(_("cannot use -a with -d"));
	}

	if (!force) {
		head_rev = lookup_commit_reference(&head_oid);
		if (!head_rev)
			die(_("Couldn't look up commit object for HEAD"));
	}
	for (i = 0; i < argc; i++, strbuf_reset(&bname)) {
		char *target = NULL;
		int flags = 0;

		strbuf_branchname(&bname, argv[i], allowed_interpret);
		free(name);
		name = mkpathdup(fmt, bname.buf);

		if (kinds == FILTER_REFS_BRANCHES) {
			const struct worktree *wt =
				find_shared_symref("HEAD", name);
			if (wt) {
				error(_("Cannot delete branch '%s' "
					"checked out at '%s'"),
				      bname.buf, wt->path);
				ret = 1;
				continue;
			}
		}

		target = resolve_refdup(name,
					RESOLVE_REF_READING
					| RESOLVE_REF_NO_RECURSE
					| RESOLVE_REF_ALLOW_BAD_NAME,
					oid.hash, &flags);
		if (!target) {
			error(remote_branch
			      ? _("remote-tracking branch '%s' not found.")
			      : _("branch '%s' not found."), bname.buf);
			ret = 1;
			continue;
		}

		if (!(flags & (REF_ISSYMREF|REF_ISBROKEN)) &&
		    check_branch_commit(bname.buf, name, &oid, head_rev, kinds,
					force)) {
			ret = 1;
			goto next;
		}

		if (delete_ref(NULL, name, is_null_oid(&oid) ? NULL : oid.hash,
			       REF_NODEREF)) {
			error(remote_branch
			      ? _("Error deleting remote-tracking branch '%s'")
			      : _("Error deleting branch '%s'"),
			      bname.buf);
			ret = 1;
			goto next;
		}
		if (!quiet) {
			printf(remote_branch
			       ? _("Deleted remote-tracking branch %s (was %s).\n")
			       : _("Deleted branch %s (was %s).\n"),
