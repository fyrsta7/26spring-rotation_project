	 * is good to deal with chains of trust, but
	 * is not consistent with what deref_tag() does
	 * which peels the onion to the core.
	 */
	return get_object(ref, 1, &obj, &oi_deref, err);
}

/*
 * Given a ref, return the value for the atom.  This lazily gets value
 * out of the object by calling populate value.
 */
static int get_ref_atom_value(struct ref_array_item *ref, int atom,
			      struct atom_value **v, struct strbuf *err)
{
	if (!ref->value) {
		if (populate_value(ref, err))
			return -1;
		fill_missing_values(ref->value);
	}
	*v = &ref->value[atom];
	return 0;
}

/*
 * Return 1 if the refname matches one of the patterns, otherwise 0.
 * A pattern can be a literal prefix (e.g. a refname "refs/heads/master"
 * matches a pattern "refs/heads/mas") or a wildcard (e.g. the same ref
 * matches "refs/heads/mas*", too).
