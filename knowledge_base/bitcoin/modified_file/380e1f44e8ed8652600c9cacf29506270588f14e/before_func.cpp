static void SipHash_32b(benchmark::Bench& bench)
{
    uint256 x;
    uint64_t k1 = 0;
    bench.run([&] {
        *((uint64_t*)x.begin()) = SipHashUint256(0, ++k1, x);
    });
}
