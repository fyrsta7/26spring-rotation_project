static bool shaderc_init(struct ra_ctx *ctx)
{
    struct priv *p = ctx->spirv->priv = talloc_zero(ctx->spirv, struct priv);

    p->compiler = shaderc_compiler_initialize();
    if (!p->compiler)
        goto error;
    p->opts = shaderc_compile_options_initialize();
    if (!p->opts)
        goto error;

    shaderc_compile_options_set_optimization_level(p->opts,
                                    shaderc_optimization_level_performance);
    if (ctx->opts.debug)
        shaderc_compile_options_set_generate_debug_info(p->opts);

    int ver, rev;
    shaderc_get_spv_version(&ver, &rev);
    ctx->spirv->compiler_version = ver * 100 + rev; // forwards compatibility
    ctx->spirv->glsl_version = 450; // impossible to query?
    return true;

error:
    shaderc_uninit(ctx);
    return false;
}
