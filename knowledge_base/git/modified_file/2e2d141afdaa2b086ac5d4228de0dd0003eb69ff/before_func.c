
#define EWAH_MASK(x) ((eword_t)1 << (x % BITS_IN_EWORD))
#define EWAH_BLOCK(x) (x / BITS_IN_EWORD)

struct bitmap *bitmap_word_alloc(size_t word_alloc)
{
	struct bitmap *bitmap = xmalloc(sizeof(struct bitmap));
	bitmap->words = xcalloc(word_alloc, sizeof(eword_t));
	bitmap->word_alloc = word_alloc;
	return bitmap;
