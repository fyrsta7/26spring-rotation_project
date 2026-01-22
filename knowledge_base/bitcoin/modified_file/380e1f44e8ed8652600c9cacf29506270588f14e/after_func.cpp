static void SipHash_32b(benchmark::Bench& bench)
{
    FastRandomContext rng{/*fDeterministic=*/true};
    auto k0{rng.rand64()}, k1{rng.rand64()};
    auto val{rng.rand256()};
    auto i{0U};
    bench.run([&] {
        ankerl::nanobench::doNotOptimizeAway(SipHashUint256(k0, k1, val));
        ++k0;
        ++k1;
        ++i;
        val.data()[i % uint256::size()] ^= i & 0xFF;
    });
}
