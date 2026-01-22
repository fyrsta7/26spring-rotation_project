    }
    return -1;
}

int StrVec::FindI(const char* s, int startAt) const {
    auto end = this->end();
    for (auto it = this->begin() + startAt; it != end; it++) {
        // TODO(perf): check length firsts
        char* s2 = *it;
        if (str::EqI(s, s2)) {
            return it.idx;
