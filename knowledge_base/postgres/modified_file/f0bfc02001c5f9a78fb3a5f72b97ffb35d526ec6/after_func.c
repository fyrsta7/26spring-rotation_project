#include <unistd.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>

#include "libpq-fe.h"
#include "libpq-int.h"
#include "libpq/libpq-fs.h"		/* must come after sys/stat.h */

#define LO_BUFSIZE		  8192

static int	lo_initialize(PGconn *conn);


/*
 * lo_open
 *	  opens an existing large object
 *
 * returns the file descriptor for use in later lo_* calls
 * return -1 upon failure.
 */
int
lo_open(PGconn *conn, Oid lobjId, int mode)
{
	int			fd;
	int			result_len;
	PQArgBlock	argv[2];
	PGresult   *res;

	argv[0].isint = 1;
	argv[0].len = 4;
	argv[0].u.integer = lobjId;
