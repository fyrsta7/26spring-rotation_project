    painter->drawRoundedRect( artRect.adjusted( 2, 2, 2, 2 ), ART_RADIUS, ART_RADIUS );
    painter->restore();

    // Draw the art pixmap
    painter->setClipPath( artRectPath );
    painter->drawPixmap( artRect, pix );
    painter->setClipping( false );

    painter->setFont( QFont( "Verdana", 7 ) );

    QRect textRect = option.rect.adjusted( 1, ART_SIZE + 2, -1, -1 );
    painter->drawText( textRect, qfu( input_item_GetTitleFbName( currentItem->inputItem() ) ),
                       QTextOption( Qt::AlignCenter ) );

}

