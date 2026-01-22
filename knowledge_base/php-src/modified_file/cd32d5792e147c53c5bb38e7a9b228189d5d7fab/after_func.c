
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#else
#include <strings.h>
#endif

static int month_tab_leap[12] = { -1, 30, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };
static int month_tab[12] =      { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };


/* Converts a Unix timestamp value into broken down time, in GMT */
void timelib_unixtime2gmt(timelib_time* tm, timelib_sll ts)
{
	timelib_sll days, remainder, tmp_days;
	timelib_sll cur_year = 1970;
	timelib_sll i;
	timelib_sll hours, minutes, seconds;
	int *months;

	days = ts / SECS_PER_DAY;
	remainder = ts - (days * SECS_PER_DAY);
	if (ts < 0 && remainder == 0) {
		days++;
		remainder -= SECS_PER_DAY;
	}
	TIMELIB_DEBUG(printf("days=%lld, rem=%lld\n", days, remainder););

	if (ts >= 0) {
		tmp_days = days + 1;

		if (tmp_days >= DAYS_PER_LYEAR_PERIOD || tmp_days <= -DAYS_PER_LYEAR_PERIOD) {
			cur_year += YEARS_PER_LYEAR_PERIOD * (tmp_days / DAYS_PER_LYEAR_PERIOD);
			tmp_days -= DAYS_PER_LYEAR_PERIOD * (tmp_days / DAYS_PER_LYEAR_PERIOD);
		}

		while (tmp_days >= DAYS_PER_LYEAR) {
			cur_year++;
			if (timelib_is_leap(cur_year)) {
				tmp_days -= DAYS_PER_LYEAR;
			} else {
				tmp_days -= DAYS_PER_YEAR;
			}
		}
	} else {
		tmp_days = days;

		/* Guess why this might be for, it has to do with a pope ;-). It's also
		 * only valid for Great Brittain and it's colonies. It needs fixing for
		 * other locales. *sigh*, why is this crap so complex! */
		/*
		if (ts <= TIMELIB_LL_CONST(-6857352000)) {
			tmp_days -= 11;
		}
		*/

		while (tmp_days <= 0) {
			if (tmp_days < -1460970) {
				cur_year -= 4000;
				TIMELIB_DEBUG(printf("tmp_days=%lld, year=%lld\n", tmp_days, cur_year););
				tmp_days += 1460970;
			} else {
				cur_year--;
				TIMELIB_DEBUG(printf("tmp_days=%lld, year=%lld\n", tmp_days, cur_year););
				if (timelib_is_leap(cur_year)) {
					tmp_days += DAYS_PER_LYEAR;
				} else {
					tmp_days += DAYS_PER_YEAR;
				}
			}
		}
		remainder += SECS_PER_DAY;
	}
	TIMELIB_DEBUG(printf("tmp_days=%lld, year=%lld\n", tmp_days, cur_year););

	months = timelib_is_leap(cur_year) ? month_tab_leap : month_tab;
	if (timelib_is_leap(cur_year) && cur_year < 1970) {
		tmp_days--;
	}
	i = 11;
	while (i > 0) {
		TIMELIB_DEBUG(printf("month=%lld (%d)\n", i, months[i]););
		if (tmp_days > months[i]) {
			break;
		}
		i--;
