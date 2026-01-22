 *      RedisModule_KeyAtPos(ctx,1);
 *      RedisModule_KeyAtPos(ctx,2);
 *  }
 *
 *  Note: in the example below the get keys API would not be needed since
 *  keys are at fixed positions. This interface is only used for commands
 *  with a more complex structure. */
void RM_KeyAtPos(RedisModuleCtx *ctx, int pos) {
    if (!(ctx->flags & REDISMODULE_CTX_KEYS_POS_REQUEST)) return;
    if (pos <= 0) return;
    ctx->keys_pos = zrealloc(ctx->keys_pos,sizeof(int)*(ctx->keys_count+1));
    ctx->keys_pos[ctx->keys_count++] = pos;
}

