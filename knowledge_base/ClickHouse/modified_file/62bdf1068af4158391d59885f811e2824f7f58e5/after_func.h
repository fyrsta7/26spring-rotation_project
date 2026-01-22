        else
            throw Exception(ErrorCodes::INCORRECT_DATA, "Unknown type of roaring bitmap");
    }

    void write(DB::WriteBuffer & out) const
    {
        UInt8 kind = isLarge() ? BitmapKind::Bitmap : BitmapKind::Small;
        writeBinary(kind, out);

        if (BitmapKind::Small == kind)
        {
            small.write(out);
        }
        else if (BitmapKind::Bitmap == kind)
        {
            std::shared_ptr<RoaringBitmap> bitmap = std::make_shared<RoaringBitmap>(*roaring_bitmap);
            bitmap->runOptimize();
            auto size = bitmap->getSizeInBytes();
            writeVarUInt(size, out);
            std::unique_ptr<char[]> buf(new char[size]);
