
    /* high frequencies */
    vlc = &huff_quad_vlc[g->count1table_select];
    last_pos=0;
    while (s_index <= 572) {
        pos = get_bits_count(&s->gb);
        if (pos >= end_pos) {
            if (pos > end_pos && last_pos){
                /* some encoders generate an incorrect size for this
                   part. We must go back into the data */
                s_index -= 4;
                init_get_bits(&s->gb, s->gb.buffer + (last_pos>>3), s->gb.size_in_bits - (last_pos&(~7)));
                skip_bits(&s->gb, last_pos&7);
            }
            break;
        }
        last_pos= pos;

        code = get_vlc2(&s->gb, vlc->table, vlc->bits, 1);
        dprintf("t=%d code=%d\n", g->count1table_select, code);
        g->sb_hybrid[s_index+0]=
        g->sb_hybrid[s_index+1]=
        g->sb_hybrid[s_index+2]=
        g->sb_hybrid[s_index+3]= 0;
        while(code){
            const static int idxtab[16]={3,3,2,2,1,1,1,1,0,0,0,0,0,0,0,0};
            int pos= s_index+idxtab[code];
            code ^= 8>>idxtab[code];
            v = l3_unscale(1, exponents[pos]);
            if(get_bits1(&s->gb))
                v = -v;
            g->sb_hybrid[pos] = v;
