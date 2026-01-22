            break;
          } else {
            px = pn;
            pn = SL_NODE_FORWARD(px, iLevel);
          }
        }

        pos[iLevel] = px;
      }
    }
  }
