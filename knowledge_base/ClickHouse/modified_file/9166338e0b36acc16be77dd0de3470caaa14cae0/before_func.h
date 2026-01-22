inline void writeEscapedString(const String & s, WriteBuffer & buf)
{
	writeAnyEscapedString<'\''>(s, buf);
}
