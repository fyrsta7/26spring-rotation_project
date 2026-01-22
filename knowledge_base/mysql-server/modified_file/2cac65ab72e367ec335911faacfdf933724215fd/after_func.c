}


static int keys_free(uchar *key, TREE_FREE mode, bulk_insert_param *param)
{
  /*
    Probably I can use info->lastkey here, but I'm not sure,
    and to be safe I'd better use local lastkey.
  */
  uchar lastkey[MI_MAX_KEY_BUFF];
  uint keylen;
  MI_KEYDEF *keyinfo;

  switch (mode) {
  case free_init:
    if (param->info->s->concurrent_insert)
    {
      rw_wrlock(&param->info->s->key_root_lock[param->keynr]);
      param->info->s->keyinfo[param->keynr].version++;
    }
    return 0;
  case free_free:
    keyinfo=param->info->s->keyinfo+param->keynr;
    keylen=_mi_keylength(keyinfo, key);
    memcpy(lastkey, key, keylen);
    return _mi_ck_write_btree(param->info,param->keynr,lastkey,
			      keylen - param->info->s->rec_reflength);
  case free_end:
    if (param->info->s->concurrent_insert)
      rw_unlock(&param->info->s->key_root_lock[param->keynr]);
    return 0;
  }
  return -1;
}


int _mi_init_bulk_insert(MI_INFO *info)
{
  MYISAM_SHARE *share=info->s;
  MI_KEYDEF *key=share->keyinfo;
  bulk_insert_param *params;
  uint i, num_keys;
  ulonglong key_map=0;

  if (info->bulk_insert)
    return 0;

  for (i=num_keys=0 ; i < share->base.keys ; i++)
  {
    if (!(key[i].flag & HA_NOSAME) && share->base.auto_key != i+1
