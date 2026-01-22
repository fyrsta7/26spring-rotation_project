
    purge_sys->offset = rec2 - page;
    purge_sys->page_no = page_get_page_no(page);
    purge_sys->iter.undo_no = trx_undo_rec_get_undo_no(rec2);
    purge_sys->iter.undo_rseg_space = space;

    if (undo_page != page) {
      /* We advance to a new page of the undo log: */
      (*n_pages_handled)++;
    }
  }

  rec_copy = trx_undo_rec_copy(rec, heap);

  mtr_commit(&mtr);

  return (rec_copy);
}

/** Fetches the next undo log record from the history list to purge. It must
 be released with the corresponding release function.
 @return copy of an undo log record or pointer to trx_purge_ignore_rec,
 if the whole undo log can skipped in purge; NULL if none left */
static MY_ATTRIBUTE((warn_unused_result))
    trx_undo_rec_t *trx_purge_fetch_next_rec(
