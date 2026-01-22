    {
        assert(n >= 0 && n <= 16);
        if (n == 0)
            return OP_0;
        return (opcodetype)(OP_1+n-1);
    }

    int FindAndDelete(const CScript& b)
    {
        int nFound = 0;
        if (b.empty())
            return nFound;
        CScript result;
        iterator pc = begin(), pc2 = begin();
        opcodetype opcode;
        do
        {
            result.insert(result.end(), pc2, pc);
            while (static_cast<size_t>(end() - pc) >= b.size() && std::equal(b.begin(), b.end(), pc))
            {
                pc = pc + b.size();
                ++nFound;
            }
            pc2 = pc;
        }
        while (GetOp(pc, opcode));

