#include <AK/Utf32View.h>

namespace AK {

inline void StringBuilder::will_append(size_t size)
{
    Checked<size_t> needed_capacity = m_length;
    needed_capacity += size;
    VERIFY(!needed_capacity.has_overflow());
    Checked<size_t> expanded_capacity = needed_capacity;
