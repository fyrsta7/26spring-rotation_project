                } else {
                    y+=base;
                }
                vec[x]=floor1_inverse_db_table[y];

                AV_DEBUG(" vec[ %d ] = %d \n", x, y);
            }

/*            for(j=1;j<hx-lx+1;++j) {  // iterating render_point
                dy=hy-ly;
                adx=hx-lx;
                ady= dy<0 ? -dy : dy;
                err=ady*j;
                off=err/adx;
                if (dy<0) {
                    predicted=ly-off;
                } else {
                    predicted=ly+off;
                }
                if (lx+j < vf->x_list[1]) {
                    vec[lx+j]=floor1_inverse_db_table[predicted];
                }
            }*/

            lx=hx;
            ly=hy;
        }
    }

    if (hx<vf->x_list[1]) {
        for(i=hx;i<vf->x_list[1];++i) {
            vec[i]=floor1_inverse_db_table[hy];
        }
    }

    AV_DEBUG(" Floor decoded\n");

    return 0;
}

// Read and decode residue

static int vorbis_residue_decode(vorbis_context *vc, vorbis_residue *vr, uint_fast8_t ch, uint_fast8_t *do_not_decode, float *vec, uint_fast16_t vlen) {
    GetBitContext *gb=&vc->gb;
    uint_fast8_t c_p_c=vc->codebooks[vr->classbook].dimensions;
    uint_fast16_t n_to_read=vr->end-vr->begin;
    uint_fast16_t ptns_to_read=n_to_read/vr->partition_size;
    uint_fast8_t classifs[ptns_to_read*vc->audio_channels];
    uint_fast8_t pass;
    uint_fast8_t ch_used;
    uint_fast8_t i,j,l;
    uint_fast16_t k;

    if (vr->type==2) {
        for(j=1;j<ch;++j) {
                do_not_decode[0]&=do_not_decode[j];  // FIXME - clobbering input
        }
        if (do_not_decode[0]) return 0;
        ch_used=1;
    } else {
        ch_used=ch;
    }

    AV_DEBUG(" residue type 0/1/2 decode begin, ch: %d  cpc %d  \n", ch, c_p_c);

    for(pass=0;pass<=vr->maxpass;++pass) { // FIXME OPTIMIZE?
        uint_fast16_t voffset;
        uint_fast16_t partition_count;
        uint_fast16_t j_times_ptns_to_read;

        voffset=vr->begin;
        for(partition_count=0;partition_count<ptns_to_read;) {  // SPEC        error
            if (!pass) {
                for(j_times_ptns_to_read=0, j=0;j<ch_used;++j) {
                    if (!do_not_decode[j]) {
                        uint_fast32_t temp=get_vlc2(gb, vc->codebooks[vr->classbook].vlc.table,
                        vc->codebooks[vr->classbook].nb_bits, 3);

                        AV_DEBUG("Classword: %d \n", temp);

                        assert(vr->classifications > 1 && temp<=65536); //needed for inverse[]
                        for(i=0;i<c_p_c;++i) {
                            uint_fast32_t temp2;

                            temp2=(((uint_fast64_t)temp) * inverse[vr->classifications])>>32;
                            if (partition_count+c_p_c-1-i < ptns_to_read) {
                                classifs[j_times_ptns_to_read+partition_count+c_p_c-1-i]=temp-temp2*vr->classifications;
                            }
                            temp=temp2;
                        }
                    }
                    j_times_ptns_to_read+=ptns_to_read;
                }
            }
            for(i=0;(i<c_p_c) && (partition_count<ptns_to_read);++i) {
                for(j_times_ptns_to_read=0, j=0;j<ch_used;++j) {
                    uint_fast16_t voffs;

                    if (!do_not_decode[j]) {
                        uint_fast8_t vqclass=classifs[j_times_ptns_to_read+partition_count];
                        int_fast16_t vqbook=vr->books[vqclass][pass];

                        if (vqbook>=0) {
                            uint_fast16_t coffs;
                            uint_fast8_t dim= vc->codebooks[vqbook].dimensions;
                            uint_fast16_t step= dim==1 ? vr->partition_size
                                              : FASTDIV(vr->partition_size, dim);
                            vorbis_codebook codebook= vc->codebooks[vqbook];

                            if (vr->type==0) {

                                voffs=voffset+j*vlen;
                                for(k=0;k<step;++k) {
                                    coffs=get_vlc2(gb, codebook.vlc.table, codebook.nb_bits, 3) * codebook.dimensions;
                                    for(l=0;l<codebook.dimensions;++l) {
                                        vec[voffs+k+l*step]+=codebook.codevectors[coffs+l];  // FPMATH
                                    }
                                }
                            }
                            else if (vr->type==1) {
                                voffs=voffset+j*vlen;
                                for(k=0;k<step;++k) {
                                    coffs=get_vlc2(gb, codebook.vlc.table, codebook.nb_bits, 3) * codebook.dimensions;
                                    for(l=0;l<codebook.dimensions;++l, ++voffs) {
                                        vec[voffs]+=codebook.codevectors[coffs+l];  // FPMATH

                                        AV_DEBUG(" pass %d offs: %d curr: %f change: %f cv offs.: %d  \n", pass, voffs, vec[voffs], codebook.codevectors[coffs+l], coffs);
                                    }
