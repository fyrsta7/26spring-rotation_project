    void loadFonts() {
        /**
         *  !!IMPORTANT!!
         *  Always load the font files in decreasing size of their glyphs. This will make the rasterize be more
         *  efficient when packing the glyphs into the font atlas and therefor make the atlas much smaller.
         */

