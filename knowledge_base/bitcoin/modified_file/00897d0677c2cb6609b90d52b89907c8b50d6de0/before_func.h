    }
};

struct CScriptWitness
{
    // Note that this encodes the data elements being pushed, rather than
    // encoding them as a CScript that pushes them.
    std::vector<std::vector<unsigned char> > stack;

    // Some compilers complain without a default constructor
    CScriptWitness() { }

    bool IsNull() const { return stack.empty(); }

    void SetNull() { stack.clear(); stack.shrink_to_fit(); }

    std::string ToString() const;
};

/** Test for OP_SUCCESSx opcodes as defined by BIP342. */
bool IsOpSuccess(const opcodetype& opcode);

bool CheckMinimalPush(const std::vector<unsigned char>& data, opcodetype opcode);
