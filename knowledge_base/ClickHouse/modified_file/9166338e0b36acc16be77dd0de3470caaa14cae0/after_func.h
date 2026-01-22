inline void writeEscapedString(const String & s, WriteBuffer & buf)
{
	/// strpbrk хорошо оптимизирована (этот if ускоряет код в 1.5 раза)
	if (NULL == strpbrk(s.data(), "\b\f\n\r\t\'\\") && strlen(s.data()) == s.size())
		writeString(s, buf);
	else
		writeAnyEscapedString<'\''>(s, buf);
}
