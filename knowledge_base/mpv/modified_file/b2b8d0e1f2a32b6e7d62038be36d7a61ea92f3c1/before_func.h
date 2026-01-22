    }
  }
//  printf("[%02X]",ds->buffer[ds->buffer_pos]);
  return ds->buffer[ds->buffer_pos++];
}
#endif

void ds_free_packs(demux_stream_t *ds);
int ds_get_packet(demux_stream_t *ds,unsigned char **start);
int ds_get_packet_pts(demux_stream_t *ds, unsigned char **start, double *pts);
int ds_get_packet_sub(demux_stream_t *ds,unsigned char **start);
