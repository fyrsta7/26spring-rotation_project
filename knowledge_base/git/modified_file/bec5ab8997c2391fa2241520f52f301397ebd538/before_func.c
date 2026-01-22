			if (c == '/')
				len = i + 1;
			i++;
		}
		if (n == 0 || len < max) {
			max = len;
			if (!max)
				break;
		}
	}
	return max;
}

/*
 * Returns a copy of the longest leading path common among all
