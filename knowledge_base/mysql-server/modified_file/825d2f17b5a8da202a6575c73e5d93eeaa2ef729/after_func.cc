}


/**
  string copy for single byte characters set when to string is shorter than
  from string.
*/

static void do_cut_string(Copy_field *copy)
{
  const CHARSET_INFO *cs= copy->from_field()->charset();
  memcpy(copy->to_ptr, copy->from_ptr, copy->to_length());

  /* Check if we loosed any important characters */
  if (cs->cset->scan(cs,
                     (char*) copy->from_ptr + copy->to_length(),
                     (char*) copy->from_ptr + copy->from_length(),
                     MY_SEQ_SPACES) < copy->from_length() - copy->to_length())
  {
    copy->to_field()->set_warning(Sql_condition::SL_WARNING,
                                  WARN_DATA_TRUNCATED, 1);
  }
}


/**
  string copy for multi byte characters set when to string is shorter than
  from string.
*/

static void do_cut_string_complex(Copy_field *copy)
{						// Shorter string field
  int well_formed_error;
  const CHARSET_INFO *cs= copy->from_field()->charset();
  const uchar *from_end= copy->from_ptr + copy->from_length();
  size_t copy_length=
    cs->cset->well_formed_len(cs,
                              (char*) copy->from_ptr,
                              (char*) from_end, 
                              copy->to_length() / cs->mbmaxlen,
                              &well_formed_error);
  if (copy->to_length() < copy_length)
    copy_length= copy->to_length();
  memcpy(copy->to_ptr, copy->from_ptr, copy_length);

  /* Check if we lost any important characters */
  if (well_formed_error ||
      cs->cset->scan(cs, (char*) copy->from_ptr + copy_length,
                     (char*) from_end,
