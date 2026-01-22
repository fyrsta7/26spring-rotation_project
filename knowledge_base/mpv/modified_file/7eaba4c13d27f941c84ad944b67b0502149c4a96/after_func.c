	    if (character<32) continue;	// skip control characters
	    charset[charset_size] = character;
	    charcodes[charset_size] = count==2 ? code : character;
	    ++charset_size;
	}
	fclose(f);
//	encoding = basename(encoding);
    }
    if (charset_size==0) ERROR("No characters to render!");
}


// general outline
void outline(
	unsigned char *s,
	unsigned char *t,
	int width,
	int height,
	int *m,
	int r,
	int mwidth) {

    int x, y;
#if 1
    for (y = 0; y<height; y++) {
	for (x = 0; x<width; x++) {
	    const int src= s[x];
	    if(src==0) continue;
#if 0 
	    if(src==255 && x>0 && y>0 && x+1<width && y+1<height
	       && s[x-1]==255 && s[x+1]==255 && s[x-width]==255 && s[x+width]==255){
		t[x + y*width]=255;
            }else
#endif
	    {
		const int x1=(x<r) ? r-x : 0;
		const int y1=(y<r) ? r-y : 0;
		const int x2=(x+r>=width ) ? r+width -x : 2*r+1;
		const int y2=(y+r>=height) ? r+height-y : 2*r+1;
		register unsigned char *dstp= t + (y1+y-r)* width + x-r;
		register int *mp  = m +  y1     *mwidth;
		int my;

		for(my= y1; my<y2; my++){
//		    unsigned char *dstp= t + (my+y-r)* width + x-r;
//		    int *mp  = m +  my     *mwidth;
		    register int mx;
		    for(mx= x1; mx<x2; mx++){
			const int tmp= (src*mp[mx] + 128)>>8;
			if(dstp[mx] < tmp) dstp[mx]= tmp;
		    }
		    dstp+=width;
		    mp+=mwidth;
		}
            }
	}
	s+= width;
    }
#else
    for (y = 0; y<height; ++y) {
	for (x = 0; x<width; ++x, ++s, ++t) {
	  //if(s[0]>=192) printf("%d\n",s[0]);
	  if(s[0]!=255){
	    unsigned max = 0;
	    unsigned *mrow = m + r;
	    unsigned char *srow = s -r*width;
	    int x1=(x<r)?-x:-r;
	    int x2=(x+r>=width)?(width-x-1):r;
	    int my;

	    for (my = -r; my<=r; ++my, srow+= width, mrow+= mwidth) {
		int mx;
		if (y+my < 0) continue;
		if (y+my >= height) break;

