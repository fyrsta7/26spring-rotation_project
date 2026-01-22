static bool jsre_init(struct sd_filter *ft)
{
    if (strcmp(ft->codec, "ass") != 0)
        return false;

    if (!ft->opts->rf_enable)
        return false;

    struct priv *p = talloc_zero(ft, struct priv);
    ft->priv = p;

    p->J = js_newstate(0, 0, JS_STRICT);
    if (!p->J) {
        MP_ERR(ft, "jsre: VM init error\n");
        return false;
    }
    talloc_set_destructor(p, destruct_priv);

    for (int n = 0; ft->opts->jsre_items && ft->opts->jsre_items[n]; n++) {
        char *item = ft->opts->jsre_items[n];

        int err = p_regcomp(p->J, p->num_regexes, item, JS_REGEXP_I | JS_REGEXP_M);
        if (err) {
            MP_ERR(ft, "jsre: %s -- '%s'\n", get_err(p->J), item);
            js_pop(p->J, 1);
            continue;
        }

        p->num_regexes += 1;
    }

    if (!p->num_regexes)
        return false;

    p->offset = sd_ass_fmt_offset(ft->event_format);
    return true;
}
