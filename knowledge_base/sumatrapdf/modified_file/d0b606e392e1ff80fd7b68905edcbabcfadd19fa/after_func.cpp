    }
    return -1;
}

int StrVec::FindI(const char* s, int startAt) const {
    int sLen = str::Leni(s);
    auto end = this->end();
    for (auto it = this->begin() + startAt; it != end; it++) {
        StrSpan s2 = it.Span();
        if (s2.Len() == sLen && str::EqI(s, s2.CStr())) {
            return it.idx;
