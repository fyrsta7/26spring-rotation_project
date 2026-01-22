  statistic_increment(ha_update_count,&LOCK_status);

  if (table->timestamp_field_type & TIMESTAMP_AUTO_SET_ON_UPDATE)
    table->timestamp_field->set_time();

  size= encode_quote(new_data);

  if (chain_append())
    DBUG_RETURN(-1);

  if (my_write(share->data_file, buffer.ptr(), size, MYF(MY_WME | MY_NABP)))
    DBUG_RETURN(-1);
  DBUG_RETURN(0);
