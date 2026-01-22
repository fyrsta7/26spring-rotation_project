                    /* final step and square */
                    p *= p*(4.f-two_cos_w*two_cos_w);
                    q *= q;
                }

                /* calculate linear floor value */
                {
                    q=exp( (
                             ( (amplitude*vf->amplitude_offset)/
                               (((1<<vf->amplitude_bits)-1) * sqrt(p+q)) )
                             - vf->amplitude_offset ) * .11512925f
                         );
                }

                /* fill vector */
                do { vec[i]=q; ++i; }while(vf->map[blockflag][i]==iter_cond);
            }
        }
    }
    else {
        /* this channel is unused */
        return 1;
    }

    AV_DEBUG(" Floor0 decoded\n");

    return 0;
}
static uint_fast8_t vorbis_floor1_decode(vorbis_context *vc, vorbis_floor_data *vfu, float *vec) {
    vorbis_floor1 * vf=&vfu->t1;
    GetBitContext *gb=&vc->gb;
    uint_fast16_t range_v[4]={ 256, 128, 86, 64 };
    uint_fast16_t range=range_v[vf->multiplier-1];
    uint_fast16_t floor1_Y[vf->x_list_dim];
    uint_fast16_t floor1_Y_final[vf->x_list_dim];
    uint_fast8_t floor1_flag[vf->x_list_dim];
    uint_fast8_t class_;
    uint_fast8_t cdim;
    uint_fast8_t cbits;
    uint_fast8_t csub;
    uint_fast8_t cval;
    int_fast16_t book;
    uint_fast16_t offset;
    uint_fast16_t i,j;
    uint_fast16_t *floor_x_sort=vf->x_list_order;
    /*u*/int_fast16_t adx, ady, off, predicted; // WTF ? dy/adx= (unsigned)dy/adx ?
    int_fast16_t dy, err;
    uint_fast16_t lx,hx, ly, hy=0;


    if (!get_bits1(gb)) return 1; // silence

// Read values (or differences) for the floor's points

    floor1_Y[0]=get_bits(gb, ilog(range-1));
    floor1_Y[1]=get_bits(gb, ilog(range-1));

    AV_DEBUG("floor 0 Y %d floor 1 Y %d \n", floor1_Y[0], floor1_Y[1]);

    offset=2;
    for(i=0;i<vf->partitions;++i) {
        class_=vf->partition_class[i];
        cdim=vf->class_dimensions[class_];
        cbits=vf->class_subclasses[class_];
        csub=(1<<cbits)-1;
        cval=0;

        AV_DEBUG("Cbits %d \n", cbits);

        if (cbits) { // this reads all subclasses for this partition's class
            cval=get_vlc2(gb, vc->codebooks[vf->class_masterbook[class_]].vlc.table,
            vc->codebooks[vf->class_masterbook[class_]].nb_bits, 3);
        }

        for(j=0;j<cdim;++j) {
            book=vf->subclass_books[class_][cval & csub];

            AV_DEBUG("book %d Cbits %d cval %d  bits:%d \n", book, cbits, cval, get_bits_count(gb));

            cval=cval>>cbits;
            if (book>0) {
                floor1_Y[offset+j]=get_vlc2(gb, vc->codebooks[book].vlc.table,
                vc->codebooks[book].nb_bits, 3);
            } else {
                floor1_Y[offset+j]=0;
            }

            AV_DEBUG(" floor(%d) = %d \n", vf->x_list[offset+j], floor1_Y[offset+j]);
        }
        offset+=cdim;
    }

// Amplitude calculation from the differences

    floor1_flag[0]=1;
    floor1_flag[1]=1;
    floor1_Y_final[0]=floor1_Y[0];
    floor1_Y_final[1]=floor1_Y[1];

    for(i=2;i<vf->x_list_dim;++i) {
        uint_fast16_t val, highroom, lowroom, room;
        uint_fast16_t high_neigh_offs;
        uint_fast16_t low_neigh_offs;

        low_neigh_offs=vf->low_neighbour[i];
        high_neigh_offs=vf->high_neighbour[i];
        dy=floor1_Y_final[high_neigh_offs]-floor1_Y_final[low_neigh_offs];  // render_point begin
        adx=vf->x_list[high_neigh_offs]-vf->x_list[low_neigh_offs];
        ady= ABS(dy);
        err=ady*(vf->x_list[i]-vf->x_list[low_neigh_offs]);
        off=err/adx;
        if (dy<0) {
            predicted=floor1_Y_final[low_neigh_offs]-off;
        } else {
            predicted=floor1_Y_final[low_neigh_offs]+off;
        } // render_point end

        val=floor1_Y[i];
        highroom=range-predicted;
        lowroom=predicted;
        if (highroom < lowroom) {
            room=highroom*2;
        } else {
            room=lowroom*2;   // SPEC mispelling
        }
        if (val) {
            floor1_flag[low_neigh_offs]=1;
            floor1_flag[high_neigh_offs]=1;
            floor1_flag[i]=1;
            if (val>=room) {
                if (highroom > lowroom) {
                    floor1_Y_final[i]=val-lowroom+predicted;
                } else {
                    floor1_Y_final[i]=predicted-val+highroom-1;
                }
            } else {
                if (val & 1) {
                    floor1_Y_final[i]=predicted-(val+1)/2;
                } else {
                    floor1_Y_final[i]=predicted+val/2;
                }
            }
        } else {
            floor1_flag[i]=0;
            floor1_Y_final[i]=predicted;
        }

        AV_DEBUG(" Decoded floor(%d) = %d / val %d \n", vf->x_list[i], floor1_Y_final[i], val);
    }

// Curve synth - connect the calculated dots and convert from dB scale FIXME optimize ?

    hx=0;
    lx=0;
    ly=floor1_Y_final[0]*vf->multiplier;  // conforms to SPEC

    vec[0]=floor1_inverse_db_table[ly];

    for(i=1;i<vf->x_list_dim;++i) {
        AV_DEBUG(" Looking at post %d \n", i);

        if (floor1_flag[floor_x_sort[i]]) {   // SPEC mispelled
            int_fast16_t x, y, dy, base, sy; // if uncommented: dy = -32 adx = 2  base = 2blablabla ?????

            hy=floor1_Y_final[floor_x_sort[i]]*vf->multiplier;
            hx=vf->x_list[floor_x_sort[i]];

            dy=hy-ly;
            adx=hx-lx;
            ady= (dy<0) ? -dy:dy;//ABS(dy);
            base=dy/adx;

            AV_DEBUG(" dy %d  adx %d base %d = %d \n", dy, adx, base, dy/adx);

            x=lx;
            y=ly;
            err=0;
            if (dy<0) {
                sy=base-1;
            } else {
                sy=base+1;
            }
            ady=ady-(base<0 ? -base : base)*adx;
            vec[x]=floor1_inverse_db_table[y];

            AV_DEBUG(" vec[ %d ] = %d \n", x, y);

            for(x=lx+1;(x<hx) && (x<vf->x_list[1]);++x) {
                err+=ady;
