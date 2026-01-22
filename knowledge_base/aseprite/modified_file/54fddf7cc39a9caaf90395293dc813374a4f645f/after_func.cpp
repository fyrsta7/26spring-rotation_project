static void ase_file_write_start_chunk(FILE* f, dio::AsepriteFrameHeader* frame_header, int type, dio::AsepriteChunk* chunk)
{
  frame_header->chunks++;

  chunk->type = type;
  chunk->start = ftell(f);

  fputl(0, f);
  fputw(0, f);
}
