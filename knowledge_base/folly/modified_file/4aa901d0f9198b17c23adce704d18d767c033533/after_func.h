        this->push(crtData, len);
        cursor.skip(len);
        return written + len;
      }

      // write the whole current IOBuf
      this->push(crtData, available);
      cursor.skip(available);
      written += available;
      len -= available;
    }
  }
};

} // namespace detail

enum class CursorAccess {
  PRIVATE,
  UNSHARE
};

template <CursorAccess access>
class RWCursor
  : public detail::CursorBase<RWCursor<access>, IOBuf>,
    public detail::Writable<RWCursor<access>> {
