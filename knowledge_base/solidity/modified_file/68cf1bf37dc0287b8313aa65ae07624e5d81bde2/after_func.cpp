	if (_other.getCategory() != getCategory())
		return false;
	MappingType const& other = dynamic_cast<MappingType const&>(_other);
	return *other.m_keyType == *m_keyType && *other.m_valueType == *m_valueType;
}

string MappingType::toString() const
{
	return "mapping(" + getKeyType()->toString() + " => " + getValueType()->toString() + ")";
}

bool TypeType::operator==(Type const& _other) const
{
	if (_other.getCategory() != getCategory())
		return false;
	TypeType const& other = dynamic_cast<TypeType const&>(_other);
	return *getActualType() == *other.getActualType();
}

MemberList const& TypeType::getMembers() const
{
	// We need to lazy-initialize it because of recursive references.
	if (!m_members)
	{
		map<string, TypePointer> members;
		if (m_actualType->getCategory() == Category::Contract && m_currentContract != nullptr)
		{
			ContractDefinition const& contract = dynamic_cast<ContractType const&>(*m_actualType).getContractDefinition();
