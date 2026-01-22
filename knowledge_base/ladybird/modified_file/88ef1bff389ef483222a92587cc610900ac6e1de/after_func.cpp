        }
    });
}

// https://html.spec.whatwg.org/multipage/canvas.html#dom-context-2d-drawimage
DOM::ExceptionOr<void> CanvasRenderingContext2D::draw_image(CanvasImageSource const& image, float destination_x, float destination_y)
{
    // If not specified, the dw and dh arguments must default to the values of sw and sh, interpreted such that one CSS pixel in the image is treated as one unit in the output bitmap's coordinate space.
    // If the sx, sy, sw, and sh arguments are omitted, then they must default to 0, 0, the image's intrinsic width in image pixels, and the image's intrinsic height in image pixels, respectively.
    // If the image has no intrinsic dimensions, then the concrete object size must be used instead, as determined using the CSS "Concrete Object Size Resolution" algorithm, with the specified size having
    // neither a definite width nor height, nor any additional constraints, the object's intrinsic properties being those of the image argument, and the default object size being the size of the output bitmap.
