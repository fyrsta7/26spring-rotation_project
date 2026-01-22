#endif

static inline void lineNoise_C(uint8_t *dst, uint8_t *src, int8_t *noise, int len, int shift){
	int i;
	noise+= shift;
	for(i=0; i<len; i++)
	{
		int v= src[i]+ noise[i];
		if(v>255) 	dst[i]=255; //FIXME optimize
		else if(v<0) 	dst[i]=0;
