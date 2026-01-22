 *  with a more complex structure. */
void RM_KeyAtPos(RedisModuleCtx *ctx, int pos) {
    if (!(ctx->flags & REDISMODULE_CTX_KEYS_POS_REQUEST)) return;
    if (pos <= 0) return;
    ctx->keys_pos = zrealloc(ctx->keys_pos,sizeof(int)*(ctx->keys_count+1));
    ctx->keys_pos[ctx->keys_count++] = pos;
}

/* Helper for RM_CreateCommand(). Truns a string representing command
 * flags into the command flags used by the Redis core.
 *
 * It returns the set of flags, or -1 if unknown flags are found. */
int commandFlagsFromString(char *s) {
    int count, j;
    int flags = 0;
    sds *tokens = sdssplitlen(s,strlen(s)," ",1,&count);
    for (j = 0; j < count; j++) {
