
		static inline void prepareScale(size_t scale, Scale & mm_scale)
		{
			Float32 fscale = static_cast<Float32>(scale);
			mm_scale = _mm_load1_ps(&fscale);
		}

		static inline void compute(const Data & in, const Scale & mm_scale, Data & out)
		{
			Float32 input[4] __attribute__((aligned(16))) = {in[0], in[1], in[2], in[3]};
			__m128 mm_value = _mm_load_ps(input);

			mm_value = _mm_mul_ps(mm_value, mm_scale);
