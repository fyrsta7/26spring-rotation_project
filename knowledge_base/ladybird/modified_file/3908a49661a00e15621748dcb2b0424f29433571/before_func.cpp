#include <AK/Utf32View.h>

namespace AK {

inline void StringBuilder::will_append(size_t size)
{
    Checked<size_t> needed_capacity = m_length;
