#include "mongo/db/exec/working_set.h"
#include "mongo/db/catalog/collection.h"
#include "mongo/util/fail_point_service.h"
#include "mongo/util/log.h"

#include "mongo/db/client.h" // XXX-ERH

namespace mongo {

    // static
    const char* CollectionScan::kStageType = "COLLSCAN";

    CollectionScan::CollectionScan(OperationContext* txn,
                                   const CollectionScanParams& params,
                                   WorkingSet* workingSet,
                                   const MatchExpression* filter)
        : _txn(txn),
          _workingSet(workingSet),
          _filter(filter),
          _params(params),
          _nsDropped(false),
          _commonStats(kStageType) {
        // Explain reports the direction of the collection scan.
        _specificStats.direction = params.direction;
    }

    PlanStage::StageState CollectionScan::work(WorkingSetID* out) {
        ++_commonStats.works;

        // Adds the amount of time taken by work() to executionTimeMillis.
        ScopedTimer timer(&_commonStats.executionTimeMillis);

        if (_nsDropped) { return PlanStage::DEAD; }

        // Do some init if we haven't already.
        if (NULL == _iter) {
            if ( _params.collection == NULL ) {
                _nsDropped = true;
                return PlanStage::DEAD;
            }

            _iter.reset( _params.collection->getIterator( _txn,
                                                          _params.start,
                                                          _params.tailable,
                                                          _params.direction ) );

            ++_commonStats.needTime;
            return PlanStage::NEED_TIME;
        }

        // What we'll return to the user.
        DiskLoc nextLoc;

        // Should we try getNext() on the underlying _iter if we're EOF?  Yes, if we're tailable.
        if (isEOF()) {
            if (!_params.tailable) {
                return PlanStage::IS_EOF;
            }
            else {
                // See if _iter gives us anything new.
                nextLoc = _iter->getNext();
                if (nextLoc.isNull()) {
                    // Nope, still EOF.
                    return PlanStage::IS_EOF;
