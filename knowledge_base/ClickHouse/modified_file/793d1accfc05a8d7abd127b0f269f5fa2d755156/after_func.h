
		static inline void prepareScale(size_t scale, Scale & mm_scale)
		{
			Float32 fscale = static_cast<Float32>(scale);
			mm_scale = _mm_load1_ps(&fscale);
		}

		static inline void compute(const Data & in, const Scale & mm_scale, Data & out)
