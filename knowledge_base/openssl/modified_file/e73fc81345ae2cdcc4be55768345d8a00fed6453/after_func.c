        X509_NAME *x509_name;
        unsigned long hash;
        int ok;

        if (ctx->type != IS_DIR) {
            ERR_raise(ERR_LIB_PROV,
                      PROV_R_SEARCH_ONLY_SUPPORTED_FOR_DIRECTORIES);
            return 0;
        }

        if (!OSSL_PARAM_get_octet_string_ptr(p, (const void **)&der, &der_len)
            || (x509_name = d2i_X509_NAME(NULL, &der, der_len)) == NULL)
            return 0;
        hash = X509_NAME_hash_ex(x509_name,
                                 ossl_prov_ctx_get0_libctx(ctx->provctx), NULL,
                                 &ok);
        BIO_snprintf(ctx->_.dir.search_name, sizeof(ctx->_.dir.search_name),
                     "%08lx", hash);
        X509_NAME_free(x509_name);
        if (ok == 0)
            return 0;
    }
    return 1;
}

/*-
 *  Loading an object from a stream
 *  -------------------------------
 */

struct file_load_data_st {
    OSSL_CALLBACK *object_cb;
    void *object_cbarg;
};

static int file_load_construct(OSSL_DECODER_INSTANCE *decoder_inst,
                               const OSSL_PARAM *params, void *construct_data)
{
    struct file_load_data_st *data = construct_data;

    /*
     * At some point, we may find it justifiable to recognise PKCS#12 and
     * handle it specially here, making |file_load()| return pass its
     * contents one piece at ta time, like |e_loader_attic.c| does.
     *
     * However, that currently means parsing them out, which converts the
     * DER encoded PKCS#12 into a bunch of EVP_PKEYs and X509s, just to
     * have to re-encode them into DER to create an object abstraction for
     * each of them.
     * It's much simpler (less churn) to pass on the object abstraction we
     * get to the load_result callback and leave it to that one to do the
     * work.  If that's libcrypto code, we know that it has much better
     * possibilities to handle the EVP_PKEYs and X509s without the extra
     * churn.
     */

    return data->object_cb(params, data->object_cbarg);
}

void file_load_cleanup(void *construct_data)
{
    /* Nothing to do */
}

static int file_setup_decoders(struct file_ctx_st *ctx)
{
    EVP_PKEY *dummy; /* for ossl_decoder_ctx_setup_for_pkey() */
    OSSL_LIB_CTX *libctx = ossl_prov_ctx_get0_libctx(ctx->provctx);
    OSSL_DECODER *to_obj = NULL; /* Last resort decoder */
    OSSL_DECODER_INSTANCE *to_obj_inst = NULL;
    OSSL_DECODER_CLEANUP *old_cleanup = NULL;
    void *old_construct_data = NULL;
    int ok = 0, expect_evp_pkey = 0;

    /* Setup for this session, so only if not already done */
    if (ctx->_.file.decoderctx == NULL) {
        if ((ctx->_.file.decoderctx = OSSL_DECODER_CTX_new()) == NULL) {
            ERR_raise(ERR_LIB_PROV, ERR_R_MALLOC_FAILURE);
            goto err;
        }
