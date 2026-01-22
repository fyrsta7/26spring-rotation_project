
    if (store == NULL)
        return;
    store->freeing = 1;
    OPENSSL_free(store->default_path);
    sk_OSSL_PROVIDER_pop_free(store->providers, provider_deactivate_free);
#ifndef FIPS_MODULE
    sk_OSSL_PROVIDER_CHILD_CB_pop_free(store->child_cbs,
                                       ossl_provider_child_cb_free);
#endif
    CRYPTO_THREAD_lock_free(store->default_path_lock);
    CRYPTO_THREAD_lock_free(store->lock);
    for (i = 0; i < store->numprovinfo; i++)
        ossl_provider_info_clear(&store->provinfo[i]);
    OPENSSL_free(store->provinfo);
    OPENSSL_free(store);
}

void *ossl_provider_store_new(OSSL_LIB_CTX *ctx)
{
    struct provider_store_st *store = OPENSSL_zalloc(sizeof(*store));

    if (store == NULL
        || (store->providers = sk_OSSL_PROVIDER_new(ossl_provider_cmp)) == NULL
        || (store->default_path_lock = CRYPTO_THREAD_lock_new()) == NULL
#ifndef FIPS_MODULE
        || (store->child_cbs = sk_OSSL_PROVIDER_CHILD_CB_new_null()) == NULL
#endif
        || (store->lock = CRYPTO_THREAD_lock_new()) == NULL) {
        ossl_provider_store_free(store);
        return NULL;
