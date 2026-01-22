    if (!_densityEstimator) {
        _densityEstimator.reset(new DensityEstimator(
            collection(), &_children, &_nearParams, _indexParams, _fullBounds));
    }

    double estimatedDistance;
    PlanStage::StageState state =
        _densityEstimator->work(expCtx(), workingSet, indexDescriptor(), out, &estimatedDistance);

    if (state == IS_EOF) {
        // We find a document in 4 neighbors at current level, but didn't at previous level.
        //
        // Assuming cell size at current level is d and data is even distributed, the distance
        // between two nearest points are at least d. The following circle with radius of 3 * d
        // covers PI * 9 * d^2, giving at most 30 documents.
        //
        // At the coarsest level, the search area is the whole earth.
        _boundsIncrement = 3 * estimatedDistance;
        invariant(_boundsIncrement > 0.0);

        // Clean up
        _densityEstimator.reset(nullptr);
    }

    return state;
}

std::unique_ptr<NearStage::CoveredInterval> GeoNear2DSphereStage::nextInterval(
    OperationContext* opCtx, WorkingSet* workingSet, const CollectionPtr& collection) {
    // The search is finished if we searched at least once and all the way to the edge
    if (_currBounds.getInner() >= 0 && _currBounds.getOuter() == _fullBounds.getOuter()) {
        return nullptr;
    }

    //
    // Setup the next interval
    //

    if (!_specificStats.intervalStats.empty()) {
        const IntervalStats& lastIntervalStats = _specificStats.intervalStats.back();

        // TODO: Generally we want small numbers of results fast, then larger numbers later
        if (lastIntervalStats.numResultsReturned < 300)
            _boundsIncrement *= 2;
        else if (lastIntervalStats.numResultsReturned > 600)
            _boundsIncrement /= 2;
    }

    invariant(_boundsIncrement > 0.0);

    R2Annulus nextBounds(_currBounds.center(),
                         _currBounds.getOuter(),
                         min(_currBounds.getOuter() + _boundsIncrement, _fullBounds.getOuter()));

    bool isLastInterval = (nextBounds.getOuter() == _fullBounds.getOuter());
    _currBounds = nextBounds;

    //
    // Setup the covering region and stages for this interval
    //

    IndexScanParams scanParams(opCtx, collection, indexDescriptor());

    // This does force us to do our own deduping of results.
    scanParams.bounds = _nearParams.baseBounds;

    // Because the planner doesn't yet set up 2D index bounds, do it ourselves here
    const string s2Field = _nearParams.nearQuery->field;
    const int s2FieldPosition = getFieldPosition(indexDescriptor(), s2Field);
    fassert(28678, s2FieldPosition >= 0);
    scanParams.bounds.fields[s2FieldPosition].intervals.clear();
    std::unique_ptr<S2Region> region(buildS2Region(_currBounds));

    std::vector<S2CellId> cover = ExpressionMapping::get2dsphereCovering(*region);

