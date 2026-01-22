  strmake(new_group_relay_log_name, rli_ptr->get_group_relay_log_name(),
          FN_REFLEN - 1);
  new_group_relay_log_pos = rli_ptr->get_group_relay_log_pos();
  /*
    Rollback positions in memory just before commit. Position values will be
    reset to their new values only on successful commit operation.
   */
  rli_ptr->set_group_master_log_name(saved_group_master_log_name);
  rli_ptr->set_group_master_log_pos(saved_group_master_log_pos);
  rli_ptr->notify_group_master_log_name_update();
  rli_ptr->set_group_relay_log_name(saved_group_relay_log_name);
  rli_ptr->set_group_relay_log_pos(saved_group_relay_log_pos);

  DBUG_PRINT("info", ("Rolling back to group master %s %llu  group relay %s"
                      " %llu\n",
                      rli_ptr->get_group_master_log_name(),
                      rli_ptr->get_group_master_log_pos(),
                      rli_ptr->get_group_relay_log_name(),
                      rli_ptr->get_group_relay_log_pos()));
  mysql_mutex_unlock(&rli_ptr->data_lock);
  error = do_commit(thd);
  mysql_mutex_lock(&rli_ptr->data_lock);
  if (error) {
    rli->report(ERROR_LEVEL, thd->get_stmt_da()->mysql_errno(),
                "Error in Xid_log_event: Commit could not be completed, '%s'",
                thd->get_stmt_da()->message_text());
  } else {
    DBUG_EXECUTE_IF(
        "crash_after_commit_before_update_pos",
        sql_print_information("Crashing "
                              "crash_after_commit_before_update_pos.");
        DBUG_SUICIDE(););
    /* Update positions on successful commit */
    rli_ptr->set_group_master_log_name(new_group_master_log_name);
    rli_ptr->set_group_master_log_pos(new_group_master_log_pos);
    rli_ptr->notify_group_master_log_name_update();
    rli_ptr->set_group_relay_log_name(new_group_relay_log_name);
    rli_ptr->set_group_relay_log_pos(new_group_relay_log_pos);

    DBUG_PRINT("info", ("Updating positions on succesful commit to group master"
                        " %s %llu  group relay %s %llu\n",
                        rli_ptr->get_group_master_log_name(),
                        rli_ptr->get_group_master_log_pos(),
                        rli_ptr->get_group_relay_log_name(),
                        rli_ptr->get_group_relay_log_pos()));

    /*
      For transactional repository the positions are flushed ahead of commit.
      Where as for non transactional rli repository the positions are flushed
      only on succesful commit.
     */
    if (!rli_ptr->is_transactional()) rli_ptr->flush_info(false);
  }
err:
  // This is Bug#24588741 fix:
  if (rli_ptr->is_group_master_log_pos_invalid)
    rli_ptr->is_group_master_log_pos_invalid = false;
  mysql_cond_broadcast(&rli_ptr->data_cond);
