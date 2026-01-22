    if( m_pImgSeq )
    {
        // Compute only the new size of an elementary image.
        // The actual resizing is done in the draw() method for now...

        // Compute the resize factors
        float factorX, factorY;
        getResizeFactors( factorX, factorY );

