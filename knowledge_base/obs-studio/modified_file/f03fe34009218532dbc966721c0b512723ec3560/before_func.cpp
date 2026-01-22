
	out[BLAKE2_HASH_LENGTH * 2] = 0;
}

void StringToHash(const wchar_t *in, BYTE *out)
{
	unsigned int temp;

	for (int i = 0; i < BLAKE2_HASH_LENGTH; i++) {
		swscanf_s(in + i * 2, L"%02x", &temp);
		out[i] = (BYTE)temp;
	}
}

bool CalculateFileHash(const wchar_t *path, BYTE *hash)
{
	blake2b_state blake2;
	if (blake2b_init(&blake2, BLAKE2_HASH_LENGTH) != 0)
		return false;

	WinHandle handle = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ,
				       nullptr, OPEN_EXISTING, 0, nullptr);
	if (handle == INVALID_HANDLE_VALUE)
		return false;

	vector<BYTE> buf;
	buf.resize(65536);

	for (;;) {
		DWORD read = 0;
		if (!ReadFile(handle, buf.data(), (DWORD)buf.size(), &read,
			      nullptr))
