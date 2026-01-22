}


SQL_SELECT::~SQL_SELECT()
{
  cleanup();
}

#undef index					// Fix for Unixware 7

QUICK_SELECT_I::QUICK_SELECT_I()
  :max_used_key_length(0),
   used_key_parts(0)
{}

QUICK_RANGE_SELECT::QUICK_RANGE_SELECT(THD *thd, TABLE *table, uint key_nr,
                                       bool no_alloc, MEM_ROOT *parent_alloc)
  :dont_free(0),error(0),free_file(0),in_range(0),cur_range(NULL),last_range(0)
{
  my_bitmap_map *bitmap;
  DBUG_ENTER("QUICK_RANGE_SELECT::QUICK_RANGE_SELECT");

  in_ror_merged_scan= 0;
  sorted= 0;
  index= key_nr;
  head=  table;
  key_part_info= head->key_info[index].key_part;
  my_init_dynamic_array(&ranges, sizeof(QUICK_RANGE*), 16, 16);

  /* 'thd' is not accessible in QUICK_RANGE_SELECT::reset(). */
  multi_range_bufsiz= thd->variables.read_rnd_buff_size;
  multi_range_count= thd->variables.multi_range_count;
  multi_range_length= 0;
  multi_range= NULL;
  multi_range_buff= NULL;

  if (!no_alloc && !parent_alloc)
  {
    // Allocates everything through the internal memroot
    init_sql_alloc(&alloc, thd->variables.range_alloc_block_size, 0);
    thd->mem_root= &alloc;
  }
  else
    bzero((char*) &alloc,sizeof(alloc));
  file= head->file;
  record= head->record[0];
  save_read_set= head->read_set;
