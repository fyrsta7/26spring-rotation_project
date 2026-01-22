 * are unloaded and later reloaded.
 *
 * The function does not take ownership of the 'cmdname' SDS string.
 * */
unsigned long ACLGetCommandID(sds cmdname) {
    sds lowername = sdsdup(cmdname);
    sdstolower(lowername);
    if (commandId == NULL) commandId = raxNew();
    void *id = raxFind(commandId,(unsigned char*)lowername,sdslen(lowername));
    if (id != raxNotFound) {
        sdsfree(lowername);
        return (unsigned long)id;
    }
    raxInsert(commandId,(unsigned char*)lowername,strlen(lowername),
              (void*)nextid,NULL);
    sdsfree(lowername);
    unsigned long thisid = nextid;
    nextid++;

    /* We never assign the last bit in the user commands bitmap structure,
     * this way we can later check if this bit is set, understanding if the
     * current ACL for the user was created starting with a +@all to add all
     * the possible commands and just subtracting other single commands or
     * categories, or if, instead, the ACL was created just adding commands
     * and command categories from scratch, not allowing future commands by
     * default (loaded via modules). This is useful when rewriting the ACLs
     * with ACL SAVE. */
    if (nextid == USER_COMMAND_BITS_COUNT-1) nextid++;
    return thisid;
}

/* Clear command id table and reset nextid to 0. */
void ACLClearCommandID(void) {
    if (commandId) raxFree(commandId);
    commandId = NULL;
    nextid = 0;
}

/* Return an username by its name, or NULL if the user does not exist. */
user *ACLGetUserByName(const char *name, size_t namelen) {
    void *myuser = raxFind(Users,(unsigned char*)name,namelen);
    if (myuser == raxNotFound) return NULL;
    return myuser;
}

/* =============================================================================
 * ACL permission checks
 * ==========================================================================*/

/* Check if the key can be accessed by the selector.
 *
 * If the selector can access the key, ACL_OK is returned, otherwise
 * ACL_DENIED_KEY is returned. */
static int ACLSelectorCheckKey(aclSelector *selector, const char *key, int keylen, int keyspec_flags) {
    /* The selector can access any key */
    if (selector->flags & SELECTOR_FLAG_ALLKEYS) return ACL_OK;

    listIter li;
    listNode *ln;
    listRewind(selector->patterns,&li);

    int key_flags = 0;
    if (keyspec_flags & CMD_KEY_ACCESS) key_flags |= ACL_READ_PERMISSION;
    if (keyspec_flags & CMD_KEY_INSERT) key_flags |= ACL_WRITE_PERMISSION;
    if (keyspec_flags & CMD_KEY_DELETE) key_flags |= ACL_WRITE_PERMISSION;
    if (keyspec_flags & CMD_KEY_UPDATE) key_flags |= ACL_WRITE_PERMISSION;

    /* Test this key against every pattern. */
    while((ln = listNext(&li))) {
        keyPattern *pattern = listNodeValue(ln);
        if ((pattern->flags & key_flags) != key_flags)
            continue;
        size_t plen = sdslen(pattern->pattern);
        if (stringmatchlen(pattern->pattern,plen,key,keylen,0))
            return ACL_OK;
    }
    return ACL_DENIED_KEY;
}

/* Checks if the provided selector selector has access specified in flags
 * to all keys in the keyspace. For example, CMD_KEY_READ access requires either
 * '%R~*', '~*', or allkeys to be granted to the selector. Returns 1 if all 
 * the access flags are satisfied with this selector or 0 otherwise.
 */
static int ACLSelectorHasUnrestrictedKeyAccess(aclSelector *selector, int flags) {
    /* The selector can access any key */
    if (selector->flags & SELECTOR_FLAG_ALLKEYS) return 1;

    listIter li;
    listNode *ln;
    listRewind(selector->patterns,&li);

    int access_flags = 0;
    if (flags & CMD_KEY_ACCESS) access_flags |= ACL_READ_PERMISSION;
    if (flags & CMD_KEY_INSERT) access_flags |= ACL_WRITE_PERMISSION;
    if (flags & CMD_KEY_DELETE) access_flags |= ACL_WRITE_PERMISSION;
