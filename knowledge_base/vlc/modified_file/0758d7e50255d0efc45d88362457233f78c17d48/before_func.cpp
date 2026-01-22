void CompositorX11UISurface::resizeFbo()
{
    if (m_rootItem && m_context->makeCurrent(this))
    {
        createFbo();
        m_context->doneCurrent();
        updateSizes();
    }
}

void CompositorX11UISurface::resizeEvent(QResizeEvent *)
