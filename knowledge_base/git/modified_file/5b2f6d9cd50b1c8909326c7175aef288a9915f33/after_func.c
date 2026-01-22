	abbrev = short_commit_name(commit);

	subject_len = find_commit_subject(out->message, &subject);

	out->subject = xmemdupz(subject, subject_len);
	out->label = xstrfmt("%s... %s", abbrev, out->subject);
	out->parent_label = xstrfmt("parent of %s", out->label);

	return 0;
}

static void free_message(struct commit *commit, struct commit_message *msg)
{
	free(msg->parent_label);
	free(msg->label);
	free(msg->subject);
	unuse_commit_buffer(commit, msg->message);
}

static void print_advice(struct repository *r, int show_hint,
			 struct replay_opts *opts)
{
