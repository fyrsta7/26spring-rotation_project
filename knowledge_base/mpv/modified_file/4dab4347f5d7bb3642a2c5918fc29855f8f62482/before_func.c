		unsigned char* src = vf->priv->planes[pl];
		unsigned char* src_next = vf->priv->planes[pl] + src_stride;
		for(i = 0; i < vf->dmpi->chroma_height; ++i)
		{
			if ((vf->priv->dirty_rows[i*2] == 1)) {
				assert(vf->priv->dirty_rows[i*2 + 1] == 1);
				for (j = 0, k = 0; j < vf->dmpi->chroma_width; ++j, k+=2) {
					unsigned val = 0;
					val += *(src + k);
					val += *(src + k + 1);
					val += *(src_next + k);
					val += *(src_next + k + 1);
					*(dst + j) = val >> 2;
				}
			}
			dst += dst_stride;
			src = src_next + src_stride;
			src_next = src + src_stride;
		}
	}
}

static void my_draw_bitmap(struct vf_instance* vf, unsigned char* bitmap, int bitmap_w, int bitmap_h, int stride, int dst_x, int dst_y, unsigned color)
{
	unsigned char y = rgba2y(color);
	unsigned char u = rgba2u(color);
	unsigned char v = rgba2v(color);
