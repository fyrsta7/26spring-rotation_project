String StringBuilder::as_string() const {
	if (string_length == 0) {
		return "";
	}

	char32_t *buffer = memnew_arr(char32_t, string_length);

	int current_position = 0;

	int godot_string_elem = 0;
	int c_string_elem = 0;

	for (uint32_t i = 0; i < appended_strings.size(); i++) {
		const int32_t str_len = appended_strings[i];

		if (str_len == -1) {
			// Godot string
			const String &s = strings[godot_string_elem];

			memcpy(buffer + current_position, s.ptr(), s.length() * sizeof(char32_t));

			current_position += s.length();

			godot_string_elem++;
		} else {
			const char *s = c_strings[c_string_elem];

			for (int32_t j = 0; j < str_len; j++) {
				buffer[current_position + j] = s[j];
			}

			current_position += str_len;

			c_string_elem++;
		}
	}

	String final_string = String(buffer, string_length);

	memdelete_arr(buffer);

	return final_string;
}
