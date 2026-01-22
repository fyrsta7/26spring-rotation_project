
#include "mongo/platform/basic.h"

#include "mongo/db/time_proof_service.h"

#include "mongo/base/status.h"
#include "mongo/db/logical_time.h"
#include "mongo/platform/random.h"

namespace mongo {

/**
 * This value defines the range of times that match the cache. It is assumed that the cluster times
 * are staying within the range so the range size is defined by the mask. This assumes that the
 * implementation has a form or high 32 bit: secs low 32 bit: increment.
 */
