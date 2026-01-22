namespace regex {

using Detail::Block;

template<typename Parser>
void Regex<Parser>::run_optimization_passes()
{
    // Rewrite fork loops as atomic groups
    // e.g. a*b -> (ATOMIC a*)b
