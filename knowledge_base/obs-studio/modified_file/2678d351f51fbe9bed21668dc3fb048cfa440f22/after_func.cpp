					nullptr, nullptr, &si, &pi);
	if (success) {
		Status(L"Installing %s...", L"Visual C++ 2019 Redistributable");

		CloseHandle(pi.hThread);
		WaitForSingleObject(pi.hProcess, INFINITE);
		CloseHandle(pi.hProcess);
	} else {
		Status(L"Update failed: Could not execute "
		       L"%s (error code %d)",
		       L"Visual C++ 2019 Redistributable", (int)GetLastError());
	}

	DeleteFile(destPath.c_str());

	waitResult = WaitForSingleObject(cancelRequested, 0);
	if (waitResult == WAIT_OBJECT_0) {
		return false;
	}

	return success;
}

extern "C" void UpdateHookFiles(void);

static bool Update(wchar_t *cmdLine)
{
	/* ------------------------------------- *
	 * Check to make sure OBS isn't running  */

	HANDLE hObsUpdateMutex =
		OpenMutexW(SYNCHRONIZE, false, L"OBSStudioUpdateMutex");
	if (hObsUpdateMutex) {
		HANDLE hWait[2];
		hWait[0] = hObsUpdateMutex;
		hWait[1] = cancelRequested;

		int i = WaitForMultipleObjects(2, hWait, false, INFINITE);

		if (i == WAIT_OBJECT_0)
			ReleaseMutex(hObsUpdateMutex);

		CloseHandle(hObsUpdateMutex);

		if (i == WAIT_OBJECT_0 + 1)
			return false;
	}

	if (!WaitForOBS())
		return false;

	/* ------------------------------------- *
	 * Init crypt stuff                      */

	CryptProvider hProvider;
	if (!CryptAcquireContext(&hProvider, nullptr, MS_ENH_RSA_AES_PROV,
				 PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
		SetDlgItemTextW(hwndMain, IDC_STATUS,
				L"Update failed: CryptAcquireContext failure");
		return false;
	}

	::hProvider = hProvider;

	/* ------------------------------------- */

	SetDlgItemTextW(hwndMain, IDC_STATUS,
			L"Searching for available updates...");

	HWND hProgress = GetDlgItem(hwndMain, IDC_PROGRESS);
	LONG_PTR style = GetWindowLongPtr(hProgress, GWL_STYLE);
	SetWindowLongPtr(hProgress, GWL_STYLE, style | PBS_MARQUEE);

	SendDlgItemMessage(hwndMain, IDC_PROGRESS, PBM_SETMARQUEE, 1, 0);

	/* ------------------------------------- *
	 * Check if updating portable build      */

	bool bIsPortable = false;

	if (cmdLine[0]) {
		int argc;
		LPWSTR *argv = CommandLineToArgvW(cmdLine, &argc);

		if (argv) {
			for (int i = 0; i < argc; i++) {
				if (wcscmp(argv[i], L"Portable") == 0) {
					bIsPortable = true;
				}
			}

			LocalFree((HLOCAL)argv);
		}
	}

	/* ------------------------------------- *
	 * Get config path                       */

	wchar_t lpAppDataPath[MAX_PATH];
	lpAppDataPath[0] = 0;

	if (bIsPortable) {
		GetCurrentDirectory(_countof(lpAppDataPath), lpAppDataPath);
		StringCbCat(lpAppDataPath, sizeof(lpAppDataPath), L"\\config");
	} else {
		DWORD ret;
		ret = GetEnvironmentVariable(L"OBS_USER_APPDATA_PATH",
					     lpAppDataPath,
					     _countof(lpAppDataPath));

		if (ret >= _countof(lpAppDataPath)) {
			Status(L"Update failed: Could not determine AppData "
			       L"location");
			return false;
		}

		if (!ret) {
			CoTaskMemPtr<wchar_t> pOut;
			HRESULT hr = SHGetKnownFolderPath(
				FOLDERID_RoamingAppData, KF_FLAG_DEFAULT,
				nullptr, &pOut);
			if (hr != S_OK) {
				Status(L"Update failed: Could not determine AppData "
				       L"location");
				return false;
			}

			StringCbCopy(lpAppDataPath, sizeof(lpAppDataPath),
				     pOut);
		}
	}

	StringCbCat(lpAppDataPath, sizeof(lpAppDataPath), L"\\obs-studio");

	/* ------------------------------------- *
	 * Get download path                     */

	wchar_t manifestPath[MAX_PATH];
	wchar_t tempDirName[MAX_PATH];

	manifestPath[0] = 0;
	tempDirName[0] = 0;

	StringCbPrintf(manifestPath, sizeof(manifestPath),
		       L"%s\\updates\\manifest.json", lpAppDataPath);
	if (!GetTempPathW(_countof(tempDirName), tempDirName)) {
		Status(L"Update failed: Failed to get temp path: %ld",
		       GetLastError());
		return false;
	}
	if (!GetTempFileNameW(tempDirName, L"obs-studio", 0, tempPath)) {
		Status(L"Update failed: Failed to create temp dir name: %ld",
		       GetLastError());
		return false;
	}

	DeleteFile(tempPath);
	CreateDirectory(tempPath, nullptr);

	/* ------------------------------------- *
	 * Load manifest file                    */

	Json root;
	{
		string manifestFile = QuickReadFile(manifestPath);
		if (manifestFile.empty()) {
			Status(L"Update failed: Couldn't load manifest file");
			return false;
		}

		string error;
		root = Json::parse(manifestFile, error);

		if (!error.empty()) {
			Status(L"Update failed: Couldn't parse update "
			       L"manifest: %S",
			       error.c_str());
			return false;
		}
	}

	if (!root.is_object()) {
		Status(L"Update failed: Invalid update manifest");
		return false;
	}

	/* ------------------------------------- *
	 * Parse current manifest update files   */

	const Json::array &packages = root["packages"].array_items();
	for (size_t i = 0; i < packages.size(); i++) {
		if (!AddPackageUpdateFiles(packages, i, tempPath)) {
			Status(L"Update failed: Failed to process update packages");
			return false;
		}
	}

	SendDlgItemMessage(hwndMain, IDC_PROGRESS, PBM_SETMARQUEE, 0, 0);
	SetWindowLongPtr(hProgress, GWL_STYLE, style);

	/* ------------------------------------- *
	 * Exit if updates already installed     */

	if (!updates.size()) {
		Status(L"All available updates are already installed.");
		SetDlgItemText(hwndMain, IDC_BUTTON, L"Launch OBS");
		return true;
	}

	/* ------------------------------------- *
	 * Check for VS2019 redistributables     */

	if (!HasVS2019Redist()) {
		if (!UpdateVS2019Redists(root)) {
			return false;
		}
	}

	/* ------------------------------------- *
	 * Generate file hash json               */

	Json::array files;

	for (update_t &update : updates) {
		wchar_t whash_string[BLAKE2_HASH_STR_LENGTH];
		char hash_string[BLAKE2_HASH_STR_LENGTH];
		char outputPath[MAX_PATH];

		if (!update.has_hash)
			continue;

		/* check hash */
		HashToString(update.my_hash, whash_string);
		if (wcscmp(whash_string, HASH_NULL) == 0)
			continue;

		if (!WideToUTF8Buf(hash_string, whash_string))
			continue;
		if (!WideToUTF8Buf(outputPath, update.basename.c_str()))
			continue;

		string package_path;
		package_path = update.packageName;
		package_path += "/";
		package_path += outputPath;

		files.emplace_back(Json::object{
			{"name", package_path},
			{"hash", hash_string},
		});
	}

	/* ------------------------------------- *
	 * Send file hashes                      */

	string newManifest;

	if (files.size() > 0) {
		string post_body;
		Json(files).dump(post_body);

		int responseCode;

		int len = (int)post_body.size();
		uLong compressSize = compressBound(len);
		string compressedJson;

		compressedJson.resize(compressSize);
		compress2((Bytef *)&compressedJson[0], &compressSize,
			  (const Bytef *)post_body.c_str(), len,
			  Z_BEST_COMPRESSION);
		compressedJson.resize(compressSize);

		bool success = !!HTTPPostData(PATCH_MANIFEST_URL,
					      (BYTE *)&compressedJson[0],
					      (int)compressedJson.size(),
					      L"Accept-Encoding: gzip",
					      &responseCode, newManifest);

		if (!success)
			return false;

		if (responseCode != 200) {
			Status(L"Update failed: HTTP/%d while trying to "
			       L"download patch manifest",
			       responseCode);
			return false;
		}
	} else {
		newManifest = "[]";
	}

	/* ------------------------------------- *
	 * Parse new manifest                    */

	string error;
	root = Json::parse(newManifest, error);
	if (!error.empty()) {
		Status(L"Update failed: Couldn't parse patch manifest: %S",
		       error.c_str());
		return false;
	}

	if (!root.is_array()) {
		Status(L"Update failed: Invalid patch manifest");
		return false;
	}

	size_t packageCount = root.array_items().size();

	for (size_t i = 0; i < packageCount; i++) {
		const Json &patch = root[i];

		if (!patch.is_object()) {
			Status(L"Update failed: Invalid patch manifest");
			return false;
		}

		const Json &name_json = patch["name"];
		const Json &hash_json = patch["hash"];
		const Json &source_json = patch["source"];
		const Json &size_json = patch["size"];

		if (!name_json.is_string())
			continue;
		if (!hash_json.is_string())
			continue;
		if (!source_json.is_string())
			continue;
		if (!size_json.is_number())
			continue;

		const string &name = name_json.string_value();
		const string &hash = hash_json.string_value();
		const string &source = source_json.string_value();
		int size = size_json.int_value();

		UpdateWithPatchIfAvailable(name.c_str(), hash.c_str(),
					   source.c_str(), size);
	}

	/* ------------------------------------- *
	 * Download Updates                      */

	if (!RunDownloadWorkers(4))
		return false;

	if ((size_t)completedUpdates != updates.size()) {
		Status(L"Update failed to download all files.");
		return false;
	}

	/* ------------------------------------- *
	 * Install updates                       */

	int updatesInstalled = 0;
	int lastPosition = 0;

	SendDlgItemMessage(hwndMain, IDC_PROGRESS, PBM_SETPOS, 0, 0);

	for (update_t &update : updates) {
		if (!UpdateFile(update)) {
			return false;
		} else {
			updatesInstalled++;
			int position = (int)(((float)updatesInstalled /
					      (float)completedUpdates) *
					     100.0f);
			if (position > lastPosition) {
				lastPosition = position;
				SendDlgItemMessage(hwndMain, IDC_PROGRESS,
						   PBM_SETPOS, position, 0);
			}
		}
	}

	/* ------------------------------------- *
	 * Install virtual camera                */

	auto runcommand = [](wchar_t *cmd) {
		STARTUPINFO si = {};
		si.cb = sizeof(si);
		si.dwFlags = STARTF_USESHOWWINDOW;
		si.wShowWindow = SW_HIDE;

		PROCESS_INFORMATION pi;
		bool success = !!CreateProcessW(nullptr, cmd, nullptr, nullptr,
						false, CREATE_NEW_CONSOLE,
						nullptr, nullptr, &si, &pi);
		if (success) {
			WaitForSingleObject(pi.hProcess, INFINITE);
			CloseHandle(pi.hThread);
			CloseHandle(pi.hProcess);
		}
	};

	if (!bIsPortable) {
		wchar_t regsvr[MAX_PATH];
		wchar_t src[MAX_PATH];
		wchar_t tmp[MAX_PATH];
		wchar_t tmp2[MAX_PATH];

		SHGetFolderPathW(nullptr, CSIDL_SYSTEM, nullptr,
				 SHGFP_TYPE_CURRENT, regsvr);
		StringCbCat(regsvr, sizeof(regsvr), L"\\regsvr32.exe");

		GetCurrentDirectoryW(_countof(src), src);
		StringCbCat(src, sizeof(src),
			    L"\\data\\obs-plugins\\win-dshow\\");
