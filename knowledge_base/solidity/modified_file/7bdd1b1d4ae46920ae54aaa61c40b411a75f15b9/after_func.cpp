		str << ")";
	}
	else
		str << " UNIQUE";
	return str.str();
}

class Rules: public boost::noncopyable
{
public:
	Rules();
	void resetMatchGroups() { m_matchGroups.clear(); }
	vector<pair<Pattern, function<Pattern()>>> rules() const { return m_rules; }

private:
	using Expression = ExpressionClasses::Expression;
	map<unsigned, Expression const*> m_matchGroups;
	vector<pair<Pattern, function<Pattern()>>> m_rules;
};

Rules::Rules()
{
	// Multiple occurences of one of these inside one rule must match the same equivalence class.
	// Constants.
	Pattern A(Push);
	Pattern B(Push);
	Pattern C(Push);
	// Anything.
	Pattern X;
	Pattern Y;
	Pattern Z;
	A.setMatchGroup(1, m_matchGroups);
	B.setMatchGroup(2, m_matchGroups);
	C.setMatchGroup(3, m_matchGroups);
	X.setMatchGroup(4, m_matchGroups);
	Y.setMatchGroup(5, m_matchGroups);
	Z.setMatchGroup(6, m_matchGroups);

	m_rules = vector<pair<Pattern, function<Pattern()>>>{
		// arithmetics on constants
		{{Instruction::ADD, {A, B}}, [=]{ return A.d() + B.d(); }},
		{{Instruction::MUL, {A, B}}, [=]{ return A.d() * B.d(); }},
		{{Instruction::SUB, {A, B}}, [=]{ return A.d() - B.d(); }},
		{{Instruction::DIV, {A, B}}, [=]{ return B.d() == 0 ? 0 : A.d() / B.d(); }},
		{{Instruction::SDIV, {A, B}}, [=]{ return B.d() == 0 ? 0 : s2u(u2s(A.d()) / u2s(B.d())); }},
		{{Instruction::MOD, {A, B}}, [=]{ return B.d() == 0 ? 0 : A.d() % B.d(); }},
		{{Instruction::SMOD, {A, B}}, [=]{ return B.d() == 0 ? 0 : s2u(u2s(A.d()) % u2s(B.d())); }},
		{{Instruction::EXP, {A, B}}, [=]{ return u256(boost::multiprecision::powm(bigint(A.d()), bigint(B.d()), bigint(1) << 256)); }},
		{{Instruction::NOT, {A}}, [=]{ return ~A.d(); }},
		{{Instruction::LT, {A, B}}, [=]() { return A.d() < B.d() ? u256(1) : 0; }},
		{{Instruction::GT, {A, B}}, [=]() -> u256 { return A.d() > B.d() ? 1 : 0; }},
		{{Instruction::SLT, {A, B}}, [=]() -> u256 { return u2s(A.d()) < u2s(B.d()) ? 1 : 0; }},
		{{Instruction::SGT, {A, B}}, [=]() -> u256 { return u2s(A.d()) > u2s(B.d()) ? 1 : 0; }},
		{{Instruction::EQ, {A, B}}, [=]() -> u256 { return A.d() == B.d() ? 1 : 0; }},
		{{Instruction::ISZERO, {A}}, [=]() -> u256 { return A.d() == 0 ? 1 : 0; }},
		{{Instruction::AND, {A, B}}, [=]{ return A.d() & B.d(); }},
		{{Instruction::OR, {A, B}}, [=]{ return A.d() | B.d(); }},
		{{Instruction::XOR, {A, B}}, [=]{ return A.d() ^ B.d(); }},
		{{Instruction::BYTE, {A, B}}, [=]{ return A.d() >= 32 ? 0 : (B.d() >> unsigned(8 * (31 - A.d()))) & 0xff; }},
		{{Instruction::ADDMOD, {A, B, C}}, [=]{ return C.d() == 0 ? 0 : u256((bigint(A.d()) + bigint(B.d())) % C.d()); }},
		{{Instruction::MULMOD, {A, B, C}}, [=]{ return C.d() == 0 ? 0 : u256((bigint(A.d()) * bigint(B.d())) % C.d()); }},
		{{Instruction::MULMOD, {A, B, C}}, [=]{ return A.d() * B.d(); }},
		{{Instruction::SIGNEXTEND, {A, B}}, [=]() -> u256 {
			if (A.d() >= 31)
				return B.d();
			unsigned testBit = unsigned(A.d()) * 8 + 7;
			u256 mask = (u256(1) << testBit) - 1;
			return u256(boost::multiprecision::bit_test(B.d(), testBit) ? B.d() | ~mask : B.d() & mask);
		}},

		// invariants involving known constants
		{{Instruction::ADD, {X, 0}}, [=]{ return X; }},
		{{Instruction::MUL, {X, 1}}, [=]{ return X; }},
		{{Instruction::DIV, {X, 1}}, [=]{ return X; }},
		{{Instruction::SDIV, {X, 1}}, [=]{ return X; }},
		{{Instruction::OR, {X, 0}}, [=]{ return X; }},
		{{Instruction::XOR, {X, 0}}, [=]{ return X; }},
		{{Instruction::AND, {X, ~u256(0)}}, [=]{ return X; }},
		{{Instruction::MUL, {X, 0}}, [=]{ return u256(0); }},
		{{Instruction::DIV, {X, 0}}, [=]{ return u256(0); }},
		{{Instruction::MOD, {X, 0}}, [=]{ return u256(0); }},
		{{Instruction::MOD, {0, X}}, [=]{ return u256(0); }},
		{{Instruction::AND, {X, 0}}, [=]{ return u256(0); }},
		{{Instruction::OR, {X, ~u256(0)}}, [=]{ return ~u256(0); }},
		// operations involving an expression and itself
		{{Instruction::AND, {X, X}}, [=]{ return X; }},
		{{Instruction::OR, {X, X}}, [=]{ return X; }},
		{{Instruction::SUB, {X, X}}, [=]{ return u256(0); }},
		{{Instruction::EQ, {X, X}}, [=]{ return u256(1); }},
		{{Instruction::LT, {X, X}}, [=]{ return u256(0); }},
		{{Instruction::SLT, {X, X}}, [=]{ return u256(0); }},
		{{Instruction::GT, {X, X}}, [=]{ return u256(0); }},
		{{Instruction::SGT, {X, X}}, [=]{ return u256(0); }},
		{{Instruction::MOD, {X, X}}, [=]{ return u256(0); }},

		{{Instruction::NOT, {{Instruction::NOT, {X}}}}, [=]{ return X; }},
	};
	// Double negation of opcodes with binary result
	for (auto const& op: vector<Instruction>{
		Instruction::EQ,
		Instruction::LT,
		Instruction::SLT,
		Instruction::GT,
		Instruction::SGT
	})
		m_rules.push_back({
			{Instruction::ISZERO, {{Instruction::ISZERO, {{op, {X, Y}}}}}},
			[=]() -> Pattern { return {op, {X, Y}}; }
		});
	m_rules.push_back({
		{Instruction::ISZERO, {{Instruction::ISZERO, {{Instruction::ISZERO, {X}}}}}},
		[=]() -> Pattern { return {Instruction::ISZERO, {X}}; }
	});
	// Associative operations
	for (auto const& opFun: vector<pair<Instruction,function<u256(u256 const&,u256 const&)>>>{
		{Instruction::ADD, plus<u256>()},
		{Instruction::MUL, multiplies<u256>()},
		{Instruction::AND, bit_and<u256>()},
		{Instruction::OR, bit_or<u256>()},
		{Instruction::XOR, bit_xor<u256>()}
	})
	{
		auto op = opFun.first;
		auto fun = opFun.second;
		// Moving constants to the outside, order matters here!
		// we need actions that return expressions (or patterns?) here, and we need also reversed rules
		// (X+A)+B -> X+(A+B)
		m_rules += vector<pair<Pattern, function<Pattern()>>>{{
			{op, {{op, {X, A}}, B}},
			[=]() -> Pattern { return {op, {X, fun(A.d(), B.d())}}; }
		}, {
		// X+(Y+A) -> (X+Y)+A
			{op, {{op, {X, A}}, Y}},
			[=]() -> Pattern { return {op, {{op, {X, Y}}, A}}; }
		}, {
		// For now, we still need explicit commutativity for the inner pattern
			{op, {{op, {A, X}}, B}},
			[=]() -> Pattern { return {op, {X, fun(A.d(), B.d())}}; }
		}, {
			{op, {{op, {A, X}}, Y}},
			[=]() -> Pattern { return {op, {{op, {X, Y}}, A}}; }
		}};
	}
	// move constants across subtractions
	m_rules += vector<pair<Pattern, function<Pattern()>>>{
		{
			// X - A -> X + (-A)
			{Instruction::SUB, {X, A}},
