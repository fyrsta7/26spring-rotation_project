		     ((uchar*) info->block.level_info[0].last_blocks+
                             block_pos * info->block.recbuffer)));
  DBUG_RETURN((uchar*) info->block.level_info[0].last_blocks+
	      block_pos*info->block.recbuffer);
}


/**
  Populate HASH_INFO structure.
  
  @param key           Pointer to a HASH_INFO key to be populated
  @param next_key      HASH_INFO next_key value
  @param ptr_to_rec    HASH_INFO ptr_to_rec value
  @param hash          HASH_INFO hash value
*/

static inline void set_hash_key(HASH_INFO *key, HASH_INFO *next_key,
                                uchar *ptr_to_rec, ulong hash)
{
  key->next_key= next_key;
  key->ptr_to_rec= ptr_to_rec;
  key->hash= hash;
}


/*
  Write a hash-key to the hash-index
  SYNOPSIS
    info     Heap table info
    keyinfo  Key info
    record   Table record to added
    recpos   Memory buffer where the table record will be stored if added 
             successfully
  NOTE
    Hash index uses HP_BLOCK structure as a 'growable array' of HASH_INFO 
    structs. Array size == number of entries in hash index.
    hp_mask(hp_rec_hashnr()) maps hash entries values to hash array positions.
    If there are several hash entries with the same hash array position P,
    they are connected in a linked list via HASH_INFO::next_key. The first 
    list element is located at position P, next elements are located at 
    positions for which there is no record that should be located at that
    position. The order of elements in the list is arbitrary.

  RETURN
    0  - OK
    -1 - Out of memory
    HA_ERR_FOUND_DUPP_KEY - Duplicate record on unique key. The record was 
    still added and the caller must call hp_delete_key for it.
*/

int hp_write_key(HP_INFO *info, HP_KEYDEF *keyinfo,
		 const uchar *record, uchar *recpos)
{
  HP_SHARE *share = info->s;
  int flag;
  ulong halfbuff,hashnr,first_index;
  uchar *ptr_to_rec= NULL, *ptr_to_rec2= NULL;
  ulong hash1= 0, hash2= 0;
  HASH_INFO *empty, *gpos= NULL, *gpos2= NULL, *pos;
  DBUG_ENTER("hp_write_key");

  flag=0;
  if (!(empty= hp_find_free_hash(share,&keyinfo->block,share->records)))
    DBUG_RETURN(-1);				/* No more memory */
  halfbuff= (long) share->blength >> 1;
  pos= hp_find_hash(&keyinfo->block,(first_index=share->records-halfbuff));
  
  /*
    We're about to add one more hash array position, with hash_mask=#records.
    The number of hash positions will change and some entries might need to 
    be relocated to the newly added position. Those entries are currently 
    members of the list that starts at #first_index position (this is 
    guaranteed by properties of hp_mask(hp_rec_hashnr(X)) mapping function)
    At #first_index position currently there may be either:
    a) An entry with hashnr != first_index. We don't need to move it.
    or
    b) A list of items with hash_mask=first_index. The list contains entries
       of 2 types:
       1) entries that should be relocated to the list that starts at new 
          position we're adding ('uppper' list)
       2) entries that should be left in the list starting at #first_index 
          position ('lower' list)
  */
  if (pos != empty)				/* If some records */
  {
    do
    {
      hashnr= pos->hash;
      if (flag == 0)
      {
        /* 
          First loop, bail out if we're dealing with case a) from above 
          comment
        */
	if (hp_mask(hashnr, share->blength, share->records) != first_index)
	  break;
      }
      /*
        flag & LOWFIND - found a record that should be put into lower position
        flag & LOWUSED - lower position occupied by the record
        Same for HIGHFIND and HIGHUSED and 'upper' position

        gpos  - ptr to last element in lower position's list
        gpos2 - ptr to last element in upper position's list

        ptr_to_rec - ptr to last entry that should go into lower list.
        ptr_to_rec2 - same for upper list.
      */
      if (!(hashnr & halfbuff))
      {						
        /* Key should be put into 'lower' list */
	if (!(flag & LOWFIND))
	{
          /* key is the first element to go into lower position */
	  if (flag & HIGHFIND)
	  {
	    flag=LOWFIND | HIGHFIND;
	    /* key shall be moved to the current empty position */
	    gpos=empty;
	    ptr_to_rec=pos->ptr_to_rec;
	    empty=pos;				/* This place is now free */
	  }
	  else
	  {
            /*
              We can only get here at first iteration: key is at 'lower' 
              position pos and should be left here.
            */
	    flag=LOWFIND | LOWUSED;
	    gpos=pos;
	    ptr_to_rec=pos->ptr_to_rec;
	  }
	}
	else
        {
          /* Already have another key for lower position */
	  if (!(flag & LOWUSED))
	  {
	    /* Change link of previous lower-list key */
            set_hash_key(gpos, pos, ptr_to_rec, hash1);
	    flag= (flag & HIGHFIND) | (LOWFIND | LOWUSED);
	  }
	  gpos=pos;
	  ptr_to_rec=pos->ptr_to_rec;
	}
	hash1= pos->hash;
      }
      else
      {
        /* key will be put into 'higher' list */
	if (!(flag & HIGHFIND))
	{
	  flag= (flag & LOWFIND) | HIGHFIND;
