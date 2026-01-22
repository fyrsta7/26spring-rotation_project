{1,3}, {5,7}, {1,7}, {5,3}, {3,1}, {7,5}, {3,5}, {7,1}, 
{0,1}, {4,5}, {0,5}, {4,1}, {2,3}, {6,7}, {2,7}, {6,3}, 
{0,3}, {4,7}, {0,7}, {4,3}, {2,1}, {6,5}, {2,5}, {6,1}, 
{1,0}, {5,4}, {1,4}, {5,0}, {3,2}, {7,6}, {3,6}, {7,2}, 
{1,2}, {5,6}, {1,6}, {5,2}, {3,0}, {7,4}, {3,4}, {7,0},
};

struct vf_priv_s {
	int log2_count;
	int qp;
	int mpeg2;
	unsigned int outfmt;
	int temp_stride;
	uint8_t *src;
	int16_t *temp;
	AVCodecContext *avctx;
	DSPContext dsp;
};

#define SHIFT 22

static inline void requantize(DCTELEM dst[64], DCTELEM src[64], int qp, uint8_t *permutation){
	int i; 
	const int qmul= qp<<1;
	const int qadd= (qp-1)|1;
	const int qinv= ((1<<(SHIFT-3)) + qmul/2)/ qmul;
	int bias= 0; //FIXME
	unsigned int threshold1, threshold2;
