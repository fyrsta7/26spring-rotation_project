static void my_draw_bitmap(struct vf_instance* vf, unsigned char* bitmap, int bitmap_w, int bitmap_h, int stride, int dst_x, int dst_y, unsigned color)
{
	unsigned char y = rgba2y(color);
	unsigned char u = rgba2u(color);
	unsigned char v = rgba2v(color);
	unsigned char opacity = 255 - _a(color);
	unsigned char *src, *dsty, *dstu, *dstv;
	int i, j;
	mp_image_t* dmpi = vf->dmpi;

	src = bitmap;
	dsty = dmpi->planes[0] + dst_x + dst_y * dmpi->stride[0];
	dstu = vf->priv->planes[1] + dst_x + dst_y * vf->priv->outw;
	dstv = vf->priv->planes[2] + dst_x + dst_y * vf->priv->outw;
	for (i = 0; i < bitmap_h; ++i) {
		for (j = 0; j < bitmap_w; ++j) {
			unsigned k = ((unsigned)src[j]) * opacity / 255;
			dsty[j] = (k*y + (255-k)*dsty[j]) / 255;
			dstu[j] = (k*u + (255-k)*dstu[j]) / 255;
			dstv[j] = (k*v + (255-k)*dstv[j]) / 255;
		}
		src += stride;
		dsty += dmpi->stride[0];
		dstu += vf->priv->outw;
		dstv += vf->priv->outw;
	} 
}