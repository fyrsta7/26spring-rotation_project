            return;
        }

        num_assets = get_bits(&s->gb, 3) + 1;
        if (num_assets > 1) {
            avpriv_request_sample(s->avctx, "Multiple DTS-HD audio assets");
