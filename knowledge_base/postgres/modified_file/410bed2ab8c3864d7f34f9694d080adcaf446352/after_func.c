	}

	return count;
}

Datum
gtrgm_consistent(PG_FUNCTION_ARGS)
{
	GISTENTRY  *entry = (GISTENTRY *) PG_GETARG_POINTER(0);
	text	   *query = PG_GETARG_TEXT_P(1);
	StrategyNumber strategy = (StrategyNumber) PG_GETARG_UINT16(2);

	/* Oid		subtype = PG_GETARG_OID(3); */
	bool	   *recheck = (bool *) PG_GETARG_POINTER(4);
	TRGM	   *key = (TRGM *) DatumGetPointer(entry->key);
	TRGM	   *qtrg;
	bool		res;
	Size		querysize = VARSIZE(query);
	gtrgm_consistent_cache *cache;

	/*
	 * We keep the extracted trigrams in cache, because trigram extraction is
	 * relatively CPU-expensive.  When trying to reuse a cached value, check
	 * strategy number not just query itself, because trigram extraction
	 * depends on strategy.
	 *
	 * The cached structure is a single palloc chunk containing the
	 * gtrgm_consistent_cache header, then the input query (starting at a
	 * MAXALIGN boundary), then the TRGM value (also starting at a MAXALIGN
	 * boundary).  However we don't try to include the regex graph (if any) in
	 * that struct.  (XXX currently, this approach can leak regex graphs
	 * across index rescans.  Not clear if that's worth fixing.)
	 */
	cache = (gtrgm_consistent_cache *) fcinfo->flinfo->fn_extra;
	if (cache == NULL ||
		cache->strategy != strategy ||
		VARSIZE(cache->query) != querysize ||
		memcmp((char *) cache->query, (char *) query, querysize) != 0)
	{
		gtrgm_consistent_cache *newcache;
		TrgmPackedGraph *graph = NULL;
		Size		qtrgsize;

		switch (strategy)
		{
			case SimilarityStrategyNumber:
				qtrg = generate_trgm(VARDATA(query),
									 querysize - VARHDRSZ);
				break;
			case ILikeStrategyNumber:
#ifndef IGNORECASE
				elog(ERROR, "cannot handle ~~* with case-sensitive trigrams");
#endif
				/* FALL THRU */
			case LikeStrategyNumber:
				qtrg = generate_wildcard_trgm(VARDATA(query),
											  querysize - VARHDRSZ);
				break;
			case RegExpICaseStrategyNumber:
#ifndef IGNORECASE
				elog(ERROR, "cannot handle ~* with case-sensitive trigrams");
#endif
				/* FALL THRU */
			case RegExpStrategyNumber:
				qtrg = createTrgmNFA(query, PG_GET_COLLATION(),
									 &graph, fcinfo->flinfo->fn_mcxt);
				/* just in case an empty array is returned ... */
				if (qtrg && ARRNELEM(qtrg) <= 0)
				{
					pfree(qtrg);
					qtrg = NULL;
				}
				break;
			default:
				elog(ERROR, "unrecognized strategy number: %d", strategy);
				qtrg = NULL;	/* keep compiler quiet */
				break;
		}

		qtrgsize = qtrg ? VARSIZE(qtrg) : 0;

		newcache = (gtrgm_consistent_cache *)
			MemoryContextAlloc(fcinfo->flinfo->fn_mcxt,
							   MAXALIGN(sizeof(gtrgm_consistent_cache)) +
							   MAXALIGN(querysize) +
							   qtrgsize);

		newcache->strategy = strategy;
		newcache->query = (text *)
			((char *) newcache + MAXALIGN(sizeof(gtrgm_consistent_cache)));
		memcpy((char *) newcache->query, (char *) query, querysize);
		if (qtrg)
		{
			newcache->trigrams = (TRGM *)
				((char *) newcache->query + MAXALIGN(querysize));
			memcpy((char *) newcache->trigrams, (char *) qtrg, qtrgsize);
			/* release qtrg in case it was made in fn_mcxt */
			pfree(qtrg);
		}
		else
			newcache->trigrams = NULL;
		newcache->graph = graph;

		if (cache)
			pfree(cache);
		fcinfo->flinfo->fn_extra = (void *) newcache;
		cache = newcache;
	}

	qtrg = cache->trigrams;

	switch (strategy)
	{
		case SimilarityStrategyNumber:
			/* Similarity search is exact */
			*recheck = false;

			if (GIST_LEAF(entry))
			{					/* all leafs contains orig trgm */
				float4		tmpsml = cnt_sml(key, qtrg);

				/* strange bug at freebsd 5.2.1 and gcc 3.3.3 */
				res = (*(int *) &tmpsml == *(int *) &trgm_limit || tmpsml > trgm_limit) ? true : false;
			}
			else if (ISALLTRUE(key))
			{					/* non-leaf contains signature */
				res = true;
			}
			else
			{					/* non-leaf contains signature */
				int32		count = cnt_sml_sign_common(qtrg, GETSIGN(key));
				int32		len = ARRNELEM(qtrg);

				if (len == 0)
					res = false;
				else
					res = (((((float8) count) / ((float8) len))) >= trgm_limit) ? true : false;
			}
			break;
		case ILikeStrategyNumber:
#ifndef IGNORECASE
			elog(ERROR, "cannot handle ~~* with case-sensitive trigrams");
#endif
			/* FALL THRU */
		case LikeStrategyNumber:
			/* Wildcard search is inexact */
			*recheck = true;

			/*
			 * Check if all the extracted trigrams can be present in child
			 * nodes.
			 */
			if (GIST_LEAF(entry))
			{					/* all leafs contains orig trgm */
				res = trgm_contained_by(qtrg, key);
			}
			else if (ISALLTRUE(key))
			{					/* non-leaf contains signature */
				res = true;
			}
			else
			{					/* non-leaf contains signature */
				int32		k,
							tmp = 0,
							len = ARRNELEM(qtrg);
				trgm	   *ptr = GETARR(qtrg);
				BITVECP		sign = GETSIGN(key);

				res = true;
				for (k = 0; k < len; k++)
				{
					CPTRGM(((char *) &tmp), ptr + k);
					if (!GETBIT(sign, HASHVAL(tmp)))
					{
						res = false;
						break;
					}
				}
			}
			break;
		case RegExpICaseStrategyNumber:
#ifndef IGNORECASE
			elog(ERROR, "cannot handle ~* with case-sensitive trigrams");
#endif
			/* FALL THRU */
		case RegExpStrategyNumber:
			/* Regexp search is inexact */
			*recheck = true;

			/* Check regex match as much as we can with available info */
			if (qtrg)
			{
				if (GIST_LEAF(entry))
				{				/* all leafs contains orig trgm */
					bool	   *check;

					check = trgm_presence_map(qtrg, key);
					res = trigramsMatchGraph(cache->graph, check);
					pfree(check);
				}
				else if (ISALLTRUE(key))
				{				/* non-leaf contains signature */
					res = true;
				}
				else
				{				/* non-leaf contains signature */
					int32		k,
								tmp = 0,
								len = ARRNELEM(qtrg);
					trgm	   *ptr = GETARR(qtrg);
					BITVECP		sign = GETSIGN(key);
					bool	   *check;

					/*
					 * GETBIT() tests may give false positives, due to limited
					 * size of the sign array.	But since trigramsMatchGraph()
					 * implements a monotone boolean function, false positives
					 * in the check array can't lead to false negative answer.
