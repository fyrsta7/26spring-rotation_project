void Selection::ExtendSelection(_In_ COORD coordBufferPos)
{
    CONSOLE_INFORMATION& gci = ServiceLocator::LocateGlobals().getConsoleInformation();
    SCREEN_INFORMATION& screenInfo = gci.GetActiveOutputBuffer();

    _allowMouseDragSelection = true;

    // ensure position is within buffer bounds. Not less than 0 and not greater than the screen buffer size.
    try
    {
        screenInfo.GetTerminalBufferSize().Clamp(coordBufferPos);
    }
    CATCH_LOG_RETURN();

    if (!IsAreaSelected())
    {
        // we should only be extending a selection that has no area yet if we're coming from mark mode.
        // if not, just return.
        if (IsMouseInitiatedSelection())
        {
            return;
        }

        // scroll if necessary to make cursor visible.
        screenInfo.MakeCursorVisible(coordBufferPos, false);

        _dwSelectionFlags |= CONSOLE_SELECTION_NOT_EMPTY;
        _srSelectionRect.Left = _srSelectionRect.Right = _coordSelectionAnchor.X;
        _srSelectionRect.Top = _srSelectionRect.Bottom = _coordSelectionAnchor.Y;

        ShowSelection();
    }
    else
    {
        // scroll if necessary to make cursor visible.
        screenInfo.MakeCursorVisible(coordBufferPos, false);
    }

    // remember previous selection rect
    SMALL_RECT srNewSelection = _srSelectionRect;

    // update selection rect
    // this adjusts the rectangle dimensions based on which way the move was requested
    // in respect to the original selection position (the anchor)
    if (coordBufferPos.X <= _coordSelectionAnchor.X)
    {
        srNewSelection.Left = coordBufferPos.X;
        srNewSelection.Right = _coordSelectionAnchor.X;
    }
    else if (coordBufferPos.X > _coordSelectionAnchor.X)
    {
        srNewSelection.Right = coordBufferPos.X;
        srNewSelection.Left = _coordSelectionAnchor.X;
    }
    if (coordBufferPos.Y <= _coordSelectionAnchor.Y)
    {
        srNewSelection.Top = coordBufferPos.Y;
        srNewSelection.Bottom = _coordSelectionAnchor.Y;
    }
    else if (coordBufferPos.Y > _coordSelectionAnchor.Y)
    {
        srNewSelection.Bottom = coordBufferPos.Y;
        srNewSelection.Top = _coordSelectionAnchor.Y;
    }

    // call special update method to modify the displayed selection in-place
    // NOTE: Using HideSelection, editing the rectangle, then ShowSelection will cause flicker.
    //_PaintUpdateSelection(&srNewSelection);
    _srSelectionRect = srNewSelection;
    _PaintSelection();

    // Fire off an event to let accessibility apps know the selection has changed.
    auto pNotifier = ServiceLocator::LocateAccessibilityNotifier();
    if (pNotifier)
    {
        pNotifier->NotifyConsoleCaretEvent(IAccessibilityNotifier::ConsoleCaretEventFlags::CaretSelection, PACKCOORD(coordBufferPos));
    }
    LOG_IF_FAILED(ServiceLocator::LocateConsoleWindow()->SignalUia(UIA_Text_TextSelectionChangedEventId));
}
