        case BitmapFormat::Indexed8:
            do_draw_scaled_bitmap<false>(*m_target, dst_rect, clipped_rect, source, src_rect, Gfx::get_pixel<BitmapFormat::Indexed8>, opacity, scaling_mode);
            break;
        default:
