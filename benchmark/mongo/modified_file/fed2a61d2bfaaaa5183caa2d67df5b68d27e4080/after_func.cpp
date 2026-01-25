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

    // Generate a covering that does not intersect with any previous coverings
    S2CellUnion coverUnion;
    coverUnion.InitSwap(&cover);
    invariant(cover.empty());
    S2CellUnion diffUnion;
    diffUnion.GetDifference(&coverUnion, &_scannedCells);
    for (const auto& cellId : diffUnion.cell_ids()) {
        if (region->MayIntersect(S2Cell(cellId))) {
            cover.push_back(cellId);
        }
    }

    // Add the cells in this covering to the _scannedCells union
    _scannedCells.Add(cover);

    OrderedIntervalList* coveredIntervals = &scanParams.bounds.fields[s2FieldPosition];
    ExpressionMapping::S2CellIdsToIntervalsWithParents(cover, _indexParams, coveredIntervals);

    auto scan = std::make_unique<IndexScan>(expCtx(), collection, scanParams, workingSet, nullptr);

    // FetchStage owns index scan
    _children.emplace_back(std::make_unique<FetchStage>(
        expCtx(), workingSet, std::move(scan), _nearParams.filter, collection));

    return std::make_unique<CoveredInterval>(
        _children.back().get(), nextBounds.getInner(), nextBounds.getOuter(), isLastInterval);
}