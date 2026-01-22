    on the safe side.

  RETURN 
    TRUE   Yes, fields can be used in partitioning index
    FALSE  Otherwise
*/

static bool fields_ok_for_partition_index(Field **pfield)
{
  if (!pfield)
    return FALSE;
  for (; (*pfield); pfield++)
  {
    enum_field_types ftype= (*pfield)->real_type();
    if (ftype == MYSQL_TYPE_ENUM || ftype == MYSQL_TYPE_GEOMETRY)
      return FALSE;
  }
  return TRUE;
}


/*
  Create partition index description and fill related info in the context
  struct

  SYNOPSIS
    create_partition_index_description()
      prune_par  INOUT Partition pruning context

  DESCRIPTION
    Create partition index description. Partition index description is:

      part_index(used_fields_list(part_expr), used_fields_list(subpart_expr))

    If partitioning/sub-partitioning uses BLOB or Geometry fields, then
    corresponding fields_list(...) is not included into index description
    and we don't perform partition pruning for partitions/subpartitions.

  RETURN
    TRUE   Out of memory or can't do partition pruning at all
    FALSE  OK
*/

static bool create_partition_index_description(PART_PRUNE_PARAM *ppar)
{
  RANGE_OPT_PARAM *range_par= &(ppar->range_param);
  partition_info *part_info= ppar->part_info;
  uint used_part_fields, used_subpart_fields;

  used_part_fields= fields_ok_for_partition_index(part_info->part_field_array) ?
                      part_info->num_part_fields : 0;
  used_subpart_fields= 
    fields_ok_for_partition_index(part_info->subpart_field_array)? 
      part_info->num_subpart_fields : 0;
  
  uint total_parts= used_part_fields + used_subpart_fields;

  ppar->ignore_part_fields= FALSE;
  ppar->part_fields=      used_part_fields;
  ppar->last_part_partno= (int)used_part_fields - 1;

  ppar->subpart_fields= used_subpart_fields;
  ppar->last_subpart_partno= 
    used_subpart_fields?(int)(used_part_fields + used_subpart_fields - 1): -1;

  if (part_info->is_sub_partitioned())
  {
    ppar->mark_full_partition_used=  mark_full_partition_used_with_parts;
    ppar->get_top_partition_id_func= part_info->get_part_partition_id;
  }
  else
  {
    ppar->mark_full_partition_used=  mark_full_partition_used_no_parts;
    ppar->get_top_partition_id_func= part_info->get_partition_id;
  }

  KEY_PART *key_part;
  MEM_ROOT *alloc= range_par->mem_root;
  if (!total_parts || 
      !(key_part= (KEY_PART*)alloc_root(alloc, sizeof(KEY_PART)*
                                               total_parts)) ||
      !(ppar->arg_stack= (SEL_ARG**)alloc_root(alloc, sizeof(SEL_ARG*)* 
                                                      total_parts)) ||
      !(ppar->is_part_keypart= (my_bool*)alloc_root(alloc, sizeof(my_bool)*
                                                           total_parts)) ||
      !(ppar->is_subpart_keypart= (my_bool*)alloc_root(alloc, sizeof(my_bool)*
                                                           total_parts)))
    return TRUE;
 
  if (ppar->subpart_fields)
  {
    my_bitmap_map *buf;
    uint32 bufsize= bitmap_buffer_size(ppar->part_info->num_subparts);
    if (!(buf= (my_bitmap_map*) alloc_root(alloc, bufsize)))
      return TRUE;
    bitmap_init(&ppar->subparts_bitmap, buf, ppar->part_info->num_subparts,
                FALSE);
  }
  range_par->key_parts= key_part;
  Field **field= (ppar->part_fields)? part_info->part_field_array :
                                           part_info->subpart_field_array;
  bool in_subpart_fields= FALSE;
  for (uint part= 0; part < total_parts; part++, key_part++)
  {
    key_part->key=          0;
    key_part->part=	    part;
    key_part->length= (uint16)(*field)->key_length();
    key_part->store_length= (uint16)get_partition_field_store_length(*field);

    DBUG_PRINT("info", ("part %u length %u store_length %u", part,
                         key_part->length, key_part->store_length));

    key_part->field=        (*field);
    key_part->image_type =  Field::itRAW;
    /* 
      We set keypart flag to 0 here as the only HA_PART_KEY_SEG is checked
      in the RangeAnalysisModule.
    */
    key_part->flag=         0;
    /* We don't set key_parts->null_bit as it will not be used */

    ppar->is_part_keypart[part]= !in_subpart_fields;
    ppar->is_subpart_keypart[part]= in_subpart_fields;

    /*
      Check if this was last field in this array, in this case we
      switch to subpartitioning fields. (This will only happens if
      there are subpartitioning fields to cater for).
    */
    if (!*(++field))
    {
      field= part_info->subpart_field_array;
      in_subpart_fields= TRUE;
    }
  }
  range_par->key_parts_end= key_part;

  DBUG_EXECUTE("info", print_partitioning_index(range_par->key_parts,
                                                range_par->key_parts_end););
  return FALSE;
}


#ifndef NDEBUG

static void print_partitioning_index(KEY_PART *parts, KEY_PART *parts_end)
{
  DBUG_ENTER("print_partitioning_index");
  DBUG_LOCK_FILE;
  fprintf(DBUG_FILE, "partitioning INDEX(");
  for (KEY_PART *p=parts; p != parts_end; p++)
  {
    fprintf(DBUG_FILE, "%s%s", p==parts?"":" ,", p->field->field_name);
  }
  fputs(");\n", DBUG_FILE);
  DBUG_UNLOCK_FILE;
  DBUG_VOID_RETURN;
}


/* Print a "c1 < keypartX < c2" - type interval into debug trace. */
static void dbug_print_segment_range(SEL_ARG *arg, KEY_PART *part)
{
  DBUG_ENTER("dbug_print_segment_range");
  DBUG_LOCK_FILE;
  if (!(arg->min_flag & NO_MIN_RANGE))
  {
    store_key_image_to_rec(part->field, arg->min_value, part->length);
    part->field->dbug_print();
    if (arg->min_flag & NEAR_MIN)
      fputs(" < ", DBUG_FILE);
    else
      fputs(" <= ", DBUG_FILE);
  }

  fprintf(DBUG_FILE, "%s", part->field->field_name);

  if (!(arg->max_flag & NO_MAX_RANGE))
  {
    if (arg->max_flag & NEAR_MAX)
      fputs(" < ", DBUG_FILE);
    else
      fputs(" <= ", DBUG_FILE);
    store_key_image_to_rec(part->field, arg->max_value, part->length);
    part->field->dbug_print();
  }
  fputs("\n", DBUG_FILE);
  DBUG_UNLOCK_FILE;
