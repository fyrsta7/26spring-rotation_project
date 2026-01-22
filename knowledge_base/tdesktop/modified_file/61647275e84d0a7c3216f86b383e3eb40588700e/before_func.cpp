	for (const auto &image : std::as_const(_sizesCache)) {
		cache.decrement(ComputeUsage(image));
	}
	_sizesCache.clear();
}

