          if (key2_shared)
          {
            SEL_ARG *cpy= new SEL_ARG(*cur_key2);   // Must make copy
            if (!cpy)
              return 0;                         // OOM
            key1= key1->insert(cpy);
            cur_key2->increment_use_count(key1->use_count+1);
          }
          else
            key1= key1->insert(cur_key2); // Will destroy key2_root
          cur_key2= next_key2;
          continue;
        }
      }
    }

    /*
      The ranges in cur_key1 and cur_key2 are overlapping:

      cur_key2:       [----------] 
      cur_key1:    [*****-----*****]

      Corollary: cur_key1.min <= cur_key2.max
    */
    if (eq_tree(cur_key1->next_key_part, cur_key2->next_key_part))
    {
      // Merge overlapping ranges with equal next_key_part
      if (cur_key1->is_same(cur_key2))
      {
        /*
          cur_key1 covers exactly the same range as cur_key2
          Use the relevant range in key1.
        */
        cur_key1->merge_flags(cur_key2);        // Copy maybe flags
        cur_key2->increment_use_count(-1);      // Free not used tree
      }
      else
      {
        SEL_ARG *last= cur_key1;
        SEL_ARG *first= cur_key1;

        /*
          Find the last range in key1 that overlaps cur_key2 and
          where all ranges first...last have the same next_key_part as
          cur_key2.

          cur_key2:  [****----------------------*******]
          key1:         [--]  [----] [---]  [-----] [xxxx]
                        ^                   ^       ^
                        first               last    different next_key_part

          Since cur_key2 covers them, the ranges between first and last
          are merged into one range by deleting first...last-1 from
          the key1 tree. In the figure, this applies to first and the
          two consecutive ranges. The range of last is then extended:
            * last.min: Set to min(cur_key2.min, first.min)
            * last.max: If there is a last->next that overlaps cur_key2 
                        (i.e., last->next has a different next_key_part):
                                        Set adjacent to last->next.min
                        Otherwise:      Set to max(cur_key2.max, last.max)

          Result:
          cur_key2:  [****----------------------*******]
                        [--]  [----] [---]                 => deleted from key1
          key1:      [**------------------------***][xxxx]
                     ^                              ^
                     cur_key1=last                  different next_key_part
        */
        while (last->next && last->next->cmp_min_to_max(cur_key2) <= 0 &&
               eq_tree(last->next->next_key_part, cur_key2->next_key_part))
        {
          /*
            last->next is covered by cur_key2 and has same next_key_part.
            last can be deleted
          */
          SEL_ARG *save=last;
          last=last->next;
          key1= key1->tree_delete(save);
        }
        // Redirect cur_key1 to last which will cover the entire range
        cur_key1= last;

        /*
          Extend last to cover the entire range of
          [min(first.min_value,cur_key2.min_value)...last.max_value].
          If this forms a full range (the range covers all possible
          values) we return no SEL_ARG RB-tree.
        */
        bool full_range= last->copy_min(first);
        if (!full_range)
          full_range= last->copy_min(cur_key2);

        if (!full_range)
        {
          if (last->next && cur_key2->cmp_max_to_min(last->next) >= 0)
          {
            /*
              This is the case:
              cur_key2:   [-------------]
              key1:     [***------]  [xxxx]
                        ^            ^
                        last         different next_key_part

              Extend range of last up to last->next:
              cur_key2:   [-------------]
              key1:     [***--------][xxxx]
            */
            last->copy_min_to_max(last->next);
          }
          else
            /*
              This is the case:
              cur_key2:   [--------*****]
              key1:     [***---------]    [xxxx]
                        ^                 ^
                        last              different next_key_part

              Extend range of last up to max(last.max, cur_key2.max):
              cur_key2:   [--------*****]
              key1:     [***----------**] [xxxx]
             */
            full_range= last->copy_max(cur_key2);
        }
        if (full_range)
        {                                       // Full range
          key1->free_tree();
          key1->type= SEL_ARG::ALWAYS;
          key2->type= SEL_ARG::ALWAYS;
          for (; cur_key2 ; cur_key2= cur_key2->next)
            cur_key2->increment_use_count(-1);  // Free not used tree
          if (key1->maybe_flag)
            return new SEL_ARG(SEL_ARG::MAYBE_KEY);
          return 0;
        }
      }
    }

    if (cmp >= 0 && cur_key1->cmp_min_to_min(cur_key2) < 0)
    {
      /*
        This is the case ("cmp>=0" means that cur_key1.max >= cur_key2.min):
        cur_key2:                [-------]
        cur_key1:         [----------*******]
      */

      if (!cur_key1->next_key_part)
      {
        /*
          cur_key1->next_key_part is empty: cut the range that
          is covered by cur_key1 from cur_key2.
          Reason: (cur_key2->next_key_part OR
          cur_key1->next_key_part) will be empty and therefore
          equal to cur_key1->next_key_part. Thus, this part of
          the cur_key2 range is completely covered by cur_key1.
        */
        if (cur_key1->cmp_max_to_max(cur_key2) >= 0)
        {
          /*
            cur_key1 covers the entire range in cur_key2.
            cur_key2:            [-------]
            cur_key1:     [-----------------]

            Move on to next range in key2
          */
          cur_key2->increment_use_count(-1); // Free not used tree
          cur_key2= cur_key2->next;
          continue;
        }
        else
        {
          /*
            This is the case:
            cur_key2:            [-------]
            cur_key1:     [---------]

            Result:
            cur_key2:                [---]
            cur_key1:     [---------]
          */
          cur_key2->copy_max_to_min(cur_key1);
          continue;
        }
      }

      /*
        The ranges are overlapping but have not been merged because
        next_key_part of cur_key1 and cur_key2 differ. 
        cur_key2:               [----]
        cur_key1:     [------------*****]

        Split cur_key1 in two where cur_key2 starts:
        cur_key2:               [----]
        key1:         [--------][--*****]
                      ^         ^
                      insert    cur_key1
      */
      SEL_ARG *new_arg= cur_key1->clone_first(cur_key2);
      if (!new_arg)
        return 0;                               // OOM
      if ((new_arg->next_key_part= cur_key1->next_key_part))
        new_arg->increment_use_count(key1->use_count+1);
      cur_key1->copy_min_to_min(cur_key2);
      key1= key1->insert(new_arg);
    } // cur_key1.min >= cur_key2.min due to this if()

    /*
      Now cur_key2.min <= cur_key1.min <= cur_key2.max:
      cur_key2:    [---------]
      cur_key1:    [****---*****]
     */
    SEL_ARG key2_cpy(*cur_key2); // Get copy we can modify
    for (;;)
    {
      if (cur_key1->cmp_min_to_min(&key2_cpy) > 0)
      {
        /*
          This is the case:
          key2_cpy:    [------------]
          key1:                 [-*****]
                                ^
                                cur_key1
                             
          Result:
          key2_cpy:             [---]
          key1:        [-------][-*****]
                       ^        ^
                       insert   cur_key1
         */
        SEL_ARG *new_arg=key2_cpy.clone_first(cur_key1);
        if (!new_arg)
          return 0; // OOM
        if ((new_arg->next_key_part=key2_cpy.next_key_part))
          new_arg->increment_use_count(key1->use_count+1);
        key1= key1->insert(new_arg);
        key2_cpy.copy_min_to_min(cur_key1);
      } 
      // Now key2_cpy.min == cur_key1.min

      if ((cmp= cur_key1->cmp_max_to_max(&key2_cpy)) <= 0)
      {
        /*
          cur_key1.max <= key2_cpy.max:
          key2_cpy:       a)  [-------]    or b)     [----]
          cur_key1:           [----]                 [----]

          Steps:

           1) Update next_key_part of cur_key1: OR it with
              key2_cpy->next_key_part.
           2) If case a: Insert range [cur_key1.max, key2_cpy.max] 
              into key1 using next_key_part of key2_cpy

           Result:
           key1:          a)  [----][-]    or b)     [----]
         */
        cur_key1->maybe_flag|= key2_cpy.maybe_flag;
        key2_cpy.increment_use_count(key1->use_count+1);
        cur_key1->next_key_part= 
          key_or(param, cur_key1->next_key_part, key2_cpy.next_key_part);

        if (!cmp)
          break;                     // case b: done with this key2 range

        // Make key2_cpy the range [cur_key1.max, key2_cpy.max]
        key2_cpy.copy_max_to_min(cur_key1);
        if (!(cur_key1= cur_key1->next))
        {
          /*
            No more ranges in key1. Insert key2_cpy and go to "end"
            label to insert remaining ranges in key2 if any.
          */
          SEL_ARG *new_key1_range= new SEL_ARG(key2_cpy);
          if (!new_key1_range)
            return 0; // OOM
          key1= key1->insert(new_key1_range);
          cur_key2= cur_key2->next;
          goto end;
        }
        if (cur_key1->cmp_min_to_max(&key2_cpy) > 0)
        {
          /*
            The next range in key1 does not overlap with key2_cpy.
            Insert this range into key1 and move on to the next range
            in key2.
          */
          SEL_ARG *new_key1_range= new SEL_ARG(key2_cpy);
          if (!new_key1_range)
            return 0;                           // OOM
          key1= key1->insert(new_key1_range);
          break;
        }
        /*
          key2_cpy overlaps with the next range in key1 and the case
          is now "cur_key2.min <= cur_key1.min <= cur_key2.max". Go back
          to for(;;) to handle this situation.
        */
        continue;
      }
      else
      {
        /*
          This is the case:
          key2_cpy:        [-------]
          cur_key1:        [------------]

          Result:
          key1:            [-------][---]
                           ^        ^
                           new_arg  cur_key1
          Steps:

           0) If cur_key1->next_key_part is empty: do nothing.
              Reason: (key2_cpy->next_key_part OR
              cur_key1->next_key_part) will be empty and
              therefore equal to cur_key1->next_key_part. Thus,
              the range in key2_cpy is completely covered by
              cur_key1
           1) Make new_arg with range [cur_key1.min, key2_cpy.max]. 
              new_arg->next_key_part is OR between next_key_part of 
              cur_key1 and key2_cpy
           2) Make cur_key1 the range [key2_cpy.max, cur_key1.max]
           3) Insert new_arg into key1
        */
        if (!cur_key1->next_key_part) // Step 0
        {
          key2_cpy.increment_use_count(-1);     // Free not used tree
          break;
        }
        SEL_ARG *new_arg= cur_key1->clone_last(&key2_cpy);
        if (!new_arg)
          return 0; // OOM
        cur_key1->copy_max_to_min(&key2_cpy);
        cur_key1->increment_use_count(key1->use_count+1);
        /* Increment key count as it may be used for next loop */
        key2_cpy.increment_use_count(1);
        new_arg->next_key_part= key_or(param, cur_key1->next_key_part,
                                       key2_cpy.next_key_part);
        key1= key1->insert(new_arg);
        break;
      }
    }
    // Move on to next range in key2
    cur_key2= cur_key2->next;                            
  }

end:
  /*
    Add key2 ranges that are non-overlapping with and higher than the
    highest range in key1.
  */
  while (cur_key2)
  {
    SEL_ARG *next= cur_key2->next;
    if (key2_shared)
    {
      SEL_ARG *key2_cpy=new SEL_ARG(*cur_key2);  // Must make copy
      if (!key2_cpy)
        return 0;
      cur_key2->increment_use_count(key1->use_count+1);
      key1= key1->insert(key2_cpy);
    }
    else
      key1= key1->insert(cur_key2);   // Will destroy key2_root
    cur_key2= next;
  }
  key1->use_count++;

  return key1;
}


/* Compare if two trees are equal */

static bool eq_tree(SEL_ARG* a,SEL_ARG *b)
{
  if (a == b)
    return 1;
  if (!a || !b || !a->is_same(b))
    return 0;
  if (a->left != &null_element && b->left != &null_element)
  {
    if (!eq_tree(a->left,b->left))
      return 0;
  }
  else if (a->left != &null_element || b->left != &null_element)
    return 0;
  if (a->right != &null_element && b->right != &null_element)
  {
    if (!eq_tree(a->right,b->right))
      return 0;
  }
  else if (a->right != &null_element || b->right != &null_element)
    return 0;
  if (a->next_key_part != b->next_key_part)
  {						// Sub range
    if (!a->next_key_part != !b->next_key_part ||
	!eq_tree(a->next_key_part, b->next_key_part))
      return 0;
  }
  return 1;
}


SEL_ARG *
SEL_ARG::insert(SEL_ARG *key)
{
  SEL_ARG *element,**UNINIT_VAR(par),*UNINIT_VAR(last_element);

  for (element= this; element != &null_element ; )
  {
    last_element=element;
    if (key->cmp_min_to_min(element) > 0)
    {
      par= &element->right; element= element->right;
    }
    else
    {
      par = &element->left; element= element->left;
    }
  }
  *par=key;
  key->parent=last_element;
	/* Link in list */
  if (par == &last_element->left)
  {
    key->next=last_element;
    if ((key->prev=last_element->prev))
      key->prev->next=key;
    last_element->prev=key;
  }
  else
  {
    if ((key->next=last_element->next))
      key->next->prev=key;
    key->prev=last_element;
    last_element->next=key;
  }
  key->left=key->right= &null_element;
  SEL_ARG *root=rb_insert(key);			// rebalance tree
  root->use_count=this->use_count;		// copy root info
  root->elements= this->elements+1;
  root->maybe_flag=this->maybe_flag;
  return root;
}


/*
** Find best key with min <= given key
** Because the call context this should never return 0 to get_range
*/

SEL_ARG *
SEL_ARG::find_range(SEL_ARG *key)
{
  SEL_ARG *element=this,*found=0;

  for (;;)
  {
    if (element == &null_element)
      return found;
    int cmp=element->cmp_min_to_min(key);
    if (cmp == 0)
