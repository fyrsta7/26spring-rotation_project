        // Make attached data to be references instead of fields.
        filesort_info->addon_fields= NULL;
        param->addon_fields= NULL;

        param->fixed_res_length= param->ref_length;
        param->set_max_compare_length(param->max_compare_length() +
                                      param->ref_length);
        param->set_max_record_length(param->max_compare_length());

        DBUG_RETURN(true);
      }
    }
  }
  DBUG_RETURN(false);
}


/**
  Read data to buffer.

  @returns
    (uint)-1 if something goes wrong
*/
static
uint read_to_buffer(IO_CACHE *fromfile,
                    Merge_chunk *merge_chunk,
                    Sort_param *param)
{
  DBUG_ENTER("read_to_buffer");
  uint rec_length= param->max_record_length();
  ha_rows count;

  const bool packed_addon_fields= param->using_packed_addons();
  const bool using_varlen_keys= param->using_varlen_keys();

  if ((count= min(merge_chunk->max_keys(), merge_chunk->rowcount())))
  {
    size_t bytes_to_read;
    if (packed_addon_fields || using_varlen_keys)
    {
      count= merge_chunk->rowcount();
      bytes_to_read=
        min(merge_chunk->buffer_size(),
            static_cast<size_t>(fromfile->end_of_file -
                                merge_chunk->file_position()));
    }
    else
      bytes_to_read= rec_length * static_cast<size_t>(count);

    DBUG_PRINT("info", ("read_to_buffer %p at file_pos %llu bytes %llu",
                        merge_chunk,
                        static_cast<ulonglong>(merge_chunk->file_position()),
                        static_cast<ulonglong>(bytes_to_read)));
    if (mysql_file_pread(fromfile->file,
                         merge_chunk->buffer_start(),
                         bytes_to_read,
                         merge_chunk->file_position(), MYF_RW))
      DBUG_RETURN((uint) -1);			/* purecov: inspected */

    size_t num_bytes_read;
    if (packed_addon_fields || using_varlen_keys)
    {
      /*
        The last record read is most likely not complete here.
        We need to loop through all the records, reading the length fields,
        and then "chop off" the final incomplete record.
       */
      uchar *record= merge_chunk->buffer_start();
      uint ix= 0;
      for (; ix < count; ++ix)
      {
        if (using_varlen_keys &&
            (record + Sort_param::size_of_varlength_field)
            >= merge_chunk->buffer_end())
          break;                                // Incomplete record.

        uchar *start_of_payload= param->get_start_of_payload(record);
        if (start_of_payload >= merge_chunk->buffer_end())
          break;                                // Incomplete record.

        if (packed_addon_fields &&
            start_of_payload + Addon_fields::size_of_length_field >=
            merge_chunk->buffer_end())
          break;                                // Incomplete record.

        const uint res_length= packed_addon_fields ?
          Addon_fields::read_addon_length(start_of_payload) :
          param->fixed_res_length;

        if (start_of_payload + res_length >= merge_chunk->buffer_end())
          break;                                // Incomplete record.
