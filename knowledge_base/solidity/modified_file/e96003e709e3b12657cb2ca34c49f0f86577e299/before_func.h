	return (S)(bigint(_a) / bigint(_b));
}

template <class S> S modWorkaround(S const& _a, S const& _b)
{
	return (S)(bigint(_a) % bigint(_b));
}

// This works around a bug fixed with Boost 1.64.
// https://www.boost.org/doc/libs/1_68_0/libs/multiprecision/doc/html/boost_multiprecision/map/hist.html#boost_multiprecision.map.hist.multiprecision_2_3_1_boost_1_64
inline u256 shlWorkaround(u256 const& _x, unsigned _amount)
{
	return u256((bigint(_x) << _amount) & u256(-1));
}

// simplificationRuleList below was split up into parts to prevent
// stack overflows in the JavaScript optimizer for emscripten builds
// that affected certain browser versions.
template <class Pattern>
std::vector<SimplificationRule<Pattern>> simplificationRuleListPart1(
	Pattern A,
	Pattern B,
	Pattern C,
	Pattern,
	Pattern
)
{
	return std::vector<SimplificationRule<Pattern>> {
		// arithmetic on constants
		{{Instruction::ADD, {A, B}}, [=]{ return A.d() + B.d(); }, false},
		{{Instruction::MUL, {A, B}}, [=]{ return A.d() * B.d(); }, false},
		{{Instruction::SUB, {A, B}}, [=]{ return A.d() - B.d(); }, false},
		{{Instruction::DIV, {A, B}}, [=]{ return B.d() == 0 ? 0 : divWorkaround(A.d(), B.d()); }, false},
		{{Instruction::SDIV, {A, B}}, [=]{ return B.d() == 0 ? 0 : s2u(divWorkaround(u2s(A.d()), u2s(B.d()))); }, false},
		{{Instruction::MOD, {A, B}}, [=]{ return B.d() == 0 ? 0 : modWorkaround(A.d(), B.d()); }, false},
		{{Instruction::SMOD, {A, B}}, [=]{ return B.d() == 0 ? 0 : s2u(modWorkaround(u2s(A.d()), u2s(B.d()))); }, false},
		{{Instruction::EXP, {A, B}}, [=]{ return u256(boost::multiprecision::powm(bigint(A.d()), bigint(B.d()), bigint(1) << 256)); }, false},
		{{Instruction::NOT, {A}}, [=]{ return ~A.d(); }, false},
		{{Instruction::LT, {A, B}}, [=]() -> u256 { return A.d() < B.d() ? 1 : 0; }, false},
		{{Instruction::GT, {A, B}}, [=]() -> u256 { return A.d() > B.d() ? 1 : 0; }, false},
		{{Instruction::SLT, {A, B}}, [=]() -> u256 { return u2s(A.d()) < u2s(B.d()) ? 1 : 0; }, false},
		{{Instruction::SGT, {A, B}}, [=]() -> u256 { return u2s(A.d()) > u2s(B.d()) ? 1 : 0; }, false},
		{{Instruction::EQ, {A, B}}, [=]() -> u256 { return A.d() == B.d() ? 1 : 0; }, false},
		{{Instruction::ISZERO, {A}}, [=]() -> u256 { return A.d() == 0 ? 1 : 0; }, false},
		{{Instruction::AND, {A, B}}, [=]{ return A.d() & B.d(); }, false},
		{{Instruction::OR, {A, B}}, [=]{ return A.d() | B.d(); }, false},
		{{Instruction::XOR, {A, B}}, [=]{ return A.d() ^ B.d(); }, false},
		{{Instruction::BYTE, {A, B}}, [=]{ return A.d() >= 32 ? 0 : (B.d() >> unsigned(8 * (31 - A.d()))) & 0xff; }, false},
		{{Instruction::ADDMOD, {A, B, C}}, [=]{ return C.d() == 0 ? 0 : u256((bigint(A.d()) + bigint(B.d())) % C.d()); }, false},
		{{Instruction::MULMOD, {A, B, C}}, [=]{ return C.d() == 0 ? 0 : u256((bigint(A.d()) * bigint(B.d())) % C.d()); }, false},
		{{Instruction::MULMOD, {A, B, C}}, [=]{ return A.d() * B.d(); }, false},
		{{Instruction::SIGNEXTEND, {A, B}}, [=]() -> u256 {
