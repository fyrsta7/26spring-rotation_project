StringView opcode_id_name(OpCodeId opcode_id);
StringView boundary_check_type_name(BoundaryCheckType);
StringView character_compare_type_name(CharacterCompareType result);

class OpCode {
public:
    OpCode() = default;
    virtual ~OpCode() = default;

    virtual OpCodeId opcode_id() const = 0;
    virtual size_t size() const = 0;
    virtual ExecutionResult execute(MatchInput const& input, MatchState& state) const = 0;

    ALWAYS_INLINE ByteCodeValueType argument(size_t offset) const
    {
        VERIFY(state().instruction_position + offset <= m_bytecode->size());
        return m_bytecode->at(state().instruction_position + 1 + offset);
    }

    ALWAYS_INLINE StringView name() const;
    static StringView name(OpCodeId);

    ALWAYS_INLINE void set_state(MatchState& state) { m_state = &state; }

    ALWAYS_INLINE void set_bytecode(ByteCode& bytecode) { m_bytecode = &bytecode; }

    ALWAYS_INLINE MatchState const& state() const
    {
        VERIFY(m_state);
        return *m_state;
    }

