          pos.m_loc = loc;
          selectNode(c_ctx, node, loc);
          continue;
        }
        // pretend we came from left child
        pos.m_dir = dir = idir;
        break;
      }
    } while (0);
    do
    {
      /* Search for a non-empty node at leaf level to scan. */
      occup = node.getOccup();
      if (unlikely(occup == 0))
      {
        jamDebug();
        ndbrequire(fromMaintReq);
        // move back to parent - see comment in treeRemoveInner
        loc = pos.m_loc = node.getLink(2);
        pos.m_dir = dir = node.getSide();
      }
      else if (dir == idir)
      {
        // coming up from left child scan current node
        jamDebug();
        pos.m_pos = idir == 0 ? Uint32(~0) : occup;
        pos.m_dir = 3;
        break;
      }
      else
      {
        ndbrequire(dir == 1 - idir);
        // coming up from right child proceed to parent
        jamDebug();
        loc = pos.m_loc = node.getLink(2);
        pos.m_dir = dir = node.getSide();
      }
      if (unlikely(dir == 2))
