                    op(p[1], p[7]); op(p[5], p[11]); op(p[5], p[7]); op(p[3], p[9]); op(p[3], p[5]);
                    op(p[7], p[9]); op(p[1], p[2]); op(p[3], p[4]); op(p[5], p[6]); op(p[7], p[8]);
                    op(p[9], p[10]); op(p[13], p[14]); op(p[12], p[13]); op(p[13], p[14]); op(p[16], p[17]);
                    op(p[15], p[16]); op(p[16], p[17]); op(p[12], p[15]); op(p[14], p[17]); op(p[14], p[15]);
                    op(p[13], p[16]); op(p[13], p[14]); op(p[15], p[16]); op(p[19], p[20]); op(p[18], p[19]);
                    op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[21], p[23]); op(p[22], p[24]);
                    op(p[22], p[23]); op(p[18], p[21]); op(p[20], p[23]); op(p[20], p[21]); op(p[19], p[22]);
                    op(p[22], p[24]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[12], p[18]);
                    op(p[16], p[22]); op(p[16], p[18]); op(p[14], p[20]); op(p[20], p[24]); op(p[14], p[16]);
                    op(p[18], p[20]); op(p[22], p[24]); op(p[13], p[19]); op(p[17], p[23]); op(p[17], p[19]);
                    op(p[15], p[21]); op(p[15], p[17]); op(p[19], p[21]); op(p[13], p[14]); op(p[15], p[16]);
                    op(p[17], p[18]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[0], p[12]);
                    op(p[8], p[20]); op(p[8], p[12]); op(p[4], p[16]); op(p[16], p[24]); op(p[12], p[16]);
                    op(p[2], p[14]); op(p[10], p[22]); op(p[10], p[14]); op(p[6], p[18]); op(p[6], p[10]);
                    op(p[10], p[12]); op(p[1], p[13]); op(p[9], p[21]); op(p[9], p[13]); op(p[5], p[17]);
                    op(p[13], p[17]); op(p[3], p[15]); op(p[11], p[23]); op(p[11], p[15]); op(p[7], p[19]);
                    op(p[7], p[11]); op(p[11], p[13]); op(p[11], p[12]);
                    dst[j] = (T)p[12];
                }

                if( limit == size.width )
                    break;

#if (CV_SIMD || CV_SIMD_SCALABLE)
                int nlanes = VTraits<typename VecOp::arg_type>::vlanes();
#else
                int nlanes = 1;
#endif
                for( ; j <= size.width - nlanes - cn*2; j += nlanes )
                {
                    VT p0 = vop.load(row[0]+j-cn*2), p5 = vop.load(row[1]+j-cn*2), p10 = vop.load(row[2]+j-cn*2), p15 = vop.load(row[3]+j-cn*2), p20 = vop.load(row[4]+j-cn*2);
                    VT p1 = vop.load(row[0]+j-cn*1), p6 = vop.load(row[1]+j-cn*1), p11 = vop.load(row[2]+j-cn*1), p16 = vop.load(row[3]+j-cn*1), p21 = vop.load(row[4]+j-cn*1);
                    VT p2 = vop.load(row[0]+j-cn*0), p7 = vop.load(row[1]+j-cn*0), p12 = vop.load(row[2]+j-cn*0), p17 = vop.load(row[3]+j-cn*0), p22 = vop.load(row[4]+j-cn*0);
                    VT p3 = vop.load(row[0]+j+cn*1), p8 = vop.load(row[1]+j+cn*1), p13 = vop.load(row[2]+j+cn*1), p18 = vop.load(row[3]+j+cn*1), p23 = vop.load(row[4]+j+cn*1);
                    VT p4 = vop.load(row[0]+j+cn*2), p9 = vop.load(row[1]+j+cn*2), p14 = vop.load(row[2]+j+cn*2), p19 = vop.load(row[3]+j+cn*2), p24 = vop.load(row[4]+j+cn*2);

                    vop(p1, p2); vop(p0, p1); vop(p1, p2); vop(p4, p5); vop(p3, p4);
                    vop(p4, p5); vop(p0, p3); vop(p2, p5); vop(p2, p3); vop(p1, p4);
                    vop(p1, p2); vop(p3, p4); vop(p7, p8); vop(p6, p7); vop(p7, p8);
                    vop(p10, p11); vop(p9, p10); vop(p10, p11); vop(p6, p9); vop(p8, p11);
                    vop(p8, p9); vop(p7, p10); vop(p7, p8); vop(p9, p10); vop(p0, p6);
                    vop(p4, p10); vop(p4, p6); vop(p2, p8); vop(p2, p4); vop(p6, p8);
                    vop(p1, p7); vop(p5, p11); vop(p5, p7); vop(p3, p9); vop(p3, p5);
                    vop(p7, p9); vop(p1, p2); vop(p3, p4); vop(p5, p6); vop(p7, p8);
                    vop(p9, p10); vop(p13, p14); vop(p12, p13); vop(p13, p14); vop(p16, p17);
                    vop(p15, p16); vop(p16, p17); vop(p12, p15); vop(p14, p17); vop(p14, p15);
                    vop(p13, p16); vop(p13, p14); vop(p15, p16); vop(p19, p20); vop(p18, p19);
