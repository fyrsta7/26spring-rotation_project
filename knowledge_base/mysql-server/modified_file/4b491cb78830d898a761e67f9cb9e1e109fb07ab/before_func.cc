    join->procedure->add();
  DBUG_RETURN(0);
}


	/* ARGSUSED */
static int
end_write(JOIN *join, JOIN_TAB *join_tab __attribute__((unused)),
	  bool end_of_records)
{
  TABLE *table=join->tmp_table;
  int error;
  DBUG_ENTER("end_write");

  if (join->thd->killed)			// Aborted by user
  {
    my_error(ER_SERVER_SHUTDOWN,MYF(0));	/* purecov: inspected */
    DBUG_RETURN(-2);				/* purecov: inspected */
  }
  if (!end_of_records)
  {
    copy_fields(&join->tmp_table_param);
    copy_funcs(join->tmp_table_param.items_to_copy);

#ifdef TO_BE_DELETED
    if (!table->uniques)			// If not unique handling
    {
      /* Copy null values from group to row */
      ORDER   *group;
      for (group=table->group ; group ; group=group->next)
      {
	Item *item= *group->item;
	if (item->maybe_null)
	{
	  Field *field=item->get_tmp_table_field();
	  field->ptr[-1]= (byte) (field->is_null() ? 1 : 0);
	}
      }
    }
#endif
    if (!join->having || join->having->val_int())
    {
      join->found_records++;
      if ((error=table->file->write_row(table->record[0])))
      {
	if (error == HA_ERR_FOUND_DUPP_KEY ||
	    error == HA_ERR_FOUND_DUPP_UNIQUE)
	  goto end;
	if (create_myisam_from_heap(join->thd, table, &join->tmp_table_param,
				    error,1))
	  DBUG_RETURN(-1);			// Not a table_is_full error
	table->uniques=0;			// To ensure rows are the same
      }
      if (++join->send_records >= join->tmp_table_param.end_write_records &&
	  join->do_send_rows)
      {
	if (!(join->select_options & OPTION_FOUND_ROWS))
	  DBUG_RETURN(-3);
	join->do_send_rows=0;
	join->unit->select_limit_cnt = HA_POS_ERROR;
	DBUG_RETURN(0);
      }
    }
  }
end:
  DBUG_RETURN(0);
}

/* Group by searching after group record and updating it if possible */
/* ARGSUSED */

static int
end_update(JOIN *join, JOIN_TAB *join_tab __attribute__((unused)),
	   bool end_of_records)
{
  TABLE *table=join->tmp_table;
  ORDER   *group;
  int	  error;
  DBUG_ENTER("end_update");

  if (end_of_records)
    DBUG_RETURN(0);
  if (join->thd->killed)			// Aborted by user
  {
    my_error(ER_SERVER_SHUTDOWN,MYF(0));	/* purecov: inspected */
    DBUG_RETURN(-2);				/* purecov: inspected */
  }

  join->found_records++;
  copy_fields(&join->tmp_table_param);		// Groups are copied twice.
  /* Make a key of group index */
  for (group=table->group ; group ; group=group->next)
  {
    Item *item= *group->item;
    item->save_org_in_field(group->field);
#ifdef EMBEDDED_LIBRARY
    join->thd->net.last_errno= 0;
#endif
    /* Store in the used key if the field was 0 */
    if (item->maybe_null)
      group->buff[-1]=item->null_value ? 1 : 0;
  }
  // table->file->index_init(0);
  if (!table->file->index_read(table->record[1],
			       join->tmp_table_param.group_buff,0,
			       HA_READ_KEY_EXACT))
  {						/* Update old record */
    restore_record(table,record[1]);
    update_tmptable_sum_func(join->sum_funcs,table);
    if ((error=table->file->update_row(table->record[1],
				       table->record[0])))
    {
      table->file->print_error(error,MYF(0));	/* purecov: inspected */
      DBUG_RETURN(-1);				/* purecov: inspected */
    }
    DBUG_RETURN(0);
  }

  /* The null bits are already set */
  KEY_PART_INFO *key_part;
  for (group=table->group,key_part=table->key_info[0].key_part;
       group ;
       group=group->next,key_part++)
    memcpy(table->record[0]+key_part->offset, group->buff, key_part->length);

  init_tmptable_sum_functions(join->sum_funcs);
  copy_funcs(join->tmp_table_param.items_to_copy);
  if ((error=table->file->write_row(table->record[0])))
  {
    if (create_myisam_from_heap(join->thd, table, &join->tmp_table_param,
				error, 0))
      DBUG_RETURN(-1);				// Not a table_is_full error
    /* Change method to update rows */
    table->file->index_init(0);
    join->join_tab[join->tables-1].next_select=end_unique_update;
  }
  join->send_records++;
  DBUG_RETURN(0);
}

/* Like end_update, but this is done with unique constraints instead of keys */

static int
end_unique_update(JOIN *join, JOIN_TAB *join_tab __attribute__((unused)),
		  bool end_of_records)
{
