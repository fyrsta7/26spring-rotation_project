/*
  Setup internal pointers inside IO_CACHE

  SYNOPSIS
    setup_io_cache()
    info		IO_CACHE handler

  NOTES
    This is called on automaticly on init or reinit of IO_CACHE
    It must be called externally if one moves or copies an IO_CACHE
    object.
*/

void setup_io_cache(IO_CACHE* info)
{
  /* Ensure that my_b_tell() and my_b_bytes_in_cache works */
  if (info->type == WRITE_CACHE)
  {
    info->current_pos= &info->write_pos;
    info->current_end= &info->write_end;
  }
  else
  {
    info->current_pos= &info->read_pos;
    info->current_end= &info->read_end;
  }
}


static void
init_functions(IO_CACHE* info)
{
  enum cache_type type= info->type;
  switch (type) {
  case READ_NET:
    /*
      Must be initialized by the caller. The problem is that
      _my_b_net_read has to be defined in sql directory because of
      the dependency on THD, and therefore cannot be visible to
      programs that link against mysys but know nothing about THD, such
      as myisamchk
    */
    break;
  case SEQ_READ_APPEND:
    info->read_function = _my_b_seq_read;
    info->write_function = 0;			/* Force a core if used */
    break;
  default:
    info->read_function =
#ifdef THREAD
                          info->share ? _my_b_read_r :
#endif
                                        _my_b_read;
    info->write_function = _my_b_write;
  }

  setup_io_cache(info);
}


/*
  Initialize an IO_CACHE object

  SYNOPSOS
    init_io_cache()
    info		cache handler to initialize
    file		File that should be associated to to the handler
			If == -1 then real_open_cached_file()
			will be called when it's time to open file.
    cachesize		Size of buffer to allocate for read/write
			If == 0 then use my_default_record_cache_size
    type		Type of cache
    seek_offset		Where cache should start reading/writing
    use_async_io	Set to 1 of we should use async_io (if avaiable)
    cache_myflags	Bitmap of differnt flags
			MY_WME | MY_FAE | MY_NABP | MY_FNABP |
			MY_DONT_CHECK_FILESIZE

  RETURN
    0  ok
    #  error
*/

int init_io_cache(IO_CACHE *info, File file, uint cachesize,
		  enum cache_type type, my_off_t seek_offset,
		  pbool use_async_io, myf cache_myflags)
{
  uint min_cache;
  my_off_t pos;
  my_off_t end_of_file= ~(my_off_t) 0;
  DBUG_ENTER("init_io_cache");
  DBUG_PRINT("enter",("cache: 0x%lx  type: %d  pos: %ld",
		      (ulong) info, (int) type, (ulong) seek_offset));

  info->file= file;
  info->type= TYPE_NOT_SET;	    /* Don't set it until mutex are created */
  info->pos_in_file= seek_offset;
  info->pre_close = info->pre_read = info->post_read = 0;
  info->arg = 0;
  info->alloced_buffer = 0;
  info->buffer=0;
  info->seek_not_done= 0;

  if (file >= 0)
  {
    pos= my_tell(file, MYF(0));
    if ((pos == (my_off_t) -1) && (my_errno == ESPIPE))
    {
      /*
         This kind of object doesn't support seek() or tell(). Don't set a
         flag that will make us again try to seek() later and fail.
      */
      info->seek_not_done= 0;
      /*
        Additionally, if we're supposed to start somewhere other than the
        the beginning of whatever this file is, then somebody made a bad
        assumption.
      */
      DBUG_ASSERT(seek_offset == 0);
    }
    else
      info->seek_not_done= test(seek_offset != pos);
  }

  info->disk_writes= 0;
#ifdef THREAD
  info->share=0;
#endif

  if (!cachesize && !(cachesize= my_default_record_cache_size))
    DBUG_RETURN(1);				/* No cache requested */
  min_cache=use_async_io ? IO_SIZE*4 : IO_SIZE*2;
  if (type == READ_CACHE || type == SEQ_READ_APPEND)
  {						/* Assume file isn't growing */
