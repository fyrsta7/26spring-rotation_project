static void ase_file_write_start_chunk(FILE* f, dio::AsepriteFrameHeader* frame_header, int type, dio::AsepriteChunk* chunk)
{
  frame_header->chunks++;

  chunk->type = type;
  chunk->start = ftell(f);

  fseek(f, chunk->start+6, SEEK_SET);
}
