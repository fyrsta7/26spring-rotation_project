template<typename ImageTraits>
static void image_scale2x_tpl(Image* dst, const Image* src, int src_w, int src_h)
{
#if 0      // TODO complete this implementation that should be faster
           // than using a lot of get/put_pixel_fast calls.
  int dst_w = src_w*2;
  int dst_h = src_h*2;

  LockImageBits<ImageTraits> dstBits(dst, Image::WriteLock, gfx::Rect(0, 0, dst_w, dst_h));
  const LockImageBits<ImageTraits> srcBits(src);

  LockImageBits<ImageTraits>::iterator dstRow0_it = dstBits.begin();
  LockImageBits<ImageTraits>::iterator dstRow1_it = dstBits.begin();
  LockImageBits<ImageTraits>::iterator dstRow0_end = dstBits.end();
  LockImageBits<ImageTraits>::iterator dstRow1_end = dstBits.end();

  // Iterators:
  //   A
  // C P B
  //   D
  //
  // These iterators are displaced through src image and are modified in this way:
  //
  // P: is the simplest one, we just start from (0, 0) to srcEnd.
  // A: starts from row 0 (so A = P in the first row), then we start
  //    again from the row 0.
  // B: It starts from (1, row) and in the last pixel we don't moved it.
  // C: It starts from (0, 0) and then it is moved with a delay.
  // D: It starts from row 1 and continues until we reach the last
  //    row, in that case we start D iterator again.
  //
  LockImageBits<ImageTraits>::const_iterator itP, itA, itB, itC, itD, savedD;
  LockImageBits<ImageTraits>::const_iterator srcEnd = srcBits.end();
  color_t P, A, B, C, D;

  // Adjust iterators
  itP = itA = itB = itC = itD = savedD = srcBits.begin();
  dstRow1_it += dst_w;
  itD += src->width();

  for (int y=0; y<src_h; ++y) {
    if (y == 1) itA = srcBits.begin();
    if (y == src_h-2) savedD = itD;
    if (y == src_h-1) itD = savedD;
    ++itB;

    for (int x=0; x<src_w; ++x) {
      ASSERT(itP != srcEnd);
      ASSERT(itA != srcEnd);
      ASSERT(itB != srcEnd);
      ASSERT(itC != srcEnd);
      ASSERT(itD != srcEnd);
      ASSERT(dstRow0_it != dstRow0_end);
      ASSERT(dstRow1_it != dstRow1_end);

      P = *itP;
      A = *itA;                 // y-1
      B = *itB;                 // x+1
      C = *itC;                 // x-1
      D = *itD;                 // y+1

      *dstRow0_it = (C == A && C != D && A != B ? A: P);
      ++dstRow0_it;
      *dstRow0_it = (A == B && A != C && B != D ? B: P);
      ++dstRow0_it;

      *dstRow1_it = (D == C && D != B && C != A ? C: P);
      ++dstRow1_it;
      *dstRow1_it = (B == D && B != A && D != C ? D: P);
      ++dstRow1_it;

      ++itP;
      ++itA;
      if (x < src_w-2) ++itB;
      if (x > 0) ++itC;
      ++itD;
    }

    // Adjust iterators for the next two rows.
    ++itB;
    ++itC;
    dstRow0_it += dst_w;
    if (y < src_h-1)
      dstRow1_it += dst_w;
  }

  // ASSERT(itP == srcEnd);
  // ASSERT(itA == srcEnd);
  // ASSERT(itB == srcEnd);
  // ASSERT(itC == srcEnd);
  // ASSERT(itD == srcEnd);
  ASSERT(dstRow0_it == dstRow0_end);
  ASSERT(dstRow1_it == dstRow1_end);
#else

#define A c[0]
#define B c[1]
#define C c[2]
#define D c[3]
#define P c[4]

  LockImageBits<ImageTraits> dstBits(dst, gfx::Rect(0, 0, src_w*2, src_h*2));
  auto dstIt = dstBits.begin();
  auto dstIt2 = dstIt;

  color_t c[5];
  for (int y=0; y<src_h; ++y) {
    dstIt2 += src_w*2;
    for (int x=0; x<src_w; ++x) {
      P = get_pixel_fast<ImageTraits>(src, x, y);
      A = (y > 0 ? get_pixel_fast<ImageTraits>(src, x, y-1): P);
      B = (x < src_w-1 ? get_pixel_fast<ImageTraits>(src, x+1, y): P);
      C = (x > 0 ? get_pixel_fast<ImageTraits>(src, x-1, y): P);
      D = (y < src_h-1 ? get_pixel_fast<ImageTraits>(src, x, y+1): P);

      *dstIt = (C == A && C != D && A != B ? A: P);
      ++dstIt;
      *dstIt = (A == B && A != C && B != D ? B: P);
      ++dstIt;

      *dstIt2 = (D == C && D != B && C != A ? C: P);
      ++dstIt2;
      *dstIt2 = (B == D && B != A && D != C ? D: P);
      ++dstIt2;
    }
    dstIt += src_w*2;
  }

#endif
}
