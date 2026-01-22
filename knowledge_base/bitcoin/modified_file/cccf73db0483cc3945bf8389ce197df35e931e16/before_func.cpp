CBloomFilter::CBloomFilter(unsigned int nElements, double nFPRate, unsigned int nTweakIn, unsigned char nFlagsIn) :
    /**
     * The ideal size for a bloom filter with a given number of elements and false positive rate is:
     * - nElements * log(fp rate) / ln(2)^2
     * We ignore filter parameters which will create a bloom filter larger than the protocol limits
     */
    vData(min((unsigned int)(-1  / LN2SQUARED * nElements * log(nFPRate)), MAX_BLOOM_FILTER_SIZE * 8) / 8),
    /**
     * The ideal number of hash functions is filter size * ln(2) / number of elements
     * Again, we ignore filter parameters which will create a bloom filter with more hash functions than the protocol limits
     * See https://en.wikipedia.org/wiki/Bloom_filter for an explanation of these formulas
