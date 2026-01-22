/// Call function on specified join map
template <typename MapsVariant, typename Func>
inline bool joinDispatch(JoinKind kind, JoinStrictness strictness, MapsVariant & maps, Func && func)
{
    return static_for<0, KINDS.size() * STRICTNESSES.size()>([&](auto ij)
    {
        // NOTE: Avoid using nested static loop as GCC and CLANG have bugs in different ways
        // See https://stackoverflow.com/questions/44386415/gcc-and-clang-disagree-about-c17-constexpr-lambda-captures
        constexpr auto i = ij / STRICTNESSES.size();
        constexpr auto j = ij % STRICTNESSES.size();
        if (kind == KINDS[i] && strictness == STRICTNESSES[j])
        {
            func(
                std::integral_constant<JoinKind, KINDS[i]>(),
                std::integral_constant<JoinStrictness, STRICTNESSES[j]>(),
                std::get<typename MapGetter<KINDS[i], STRICTNESSES[j]>::Map>(maps));
            return true;
        }
        return false;
    });
