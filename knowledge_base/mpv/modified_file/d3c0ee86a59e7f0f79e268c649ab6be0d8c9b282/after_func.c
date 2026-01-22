	case 8: byte_len = 1; break;
	default:
	case 15:
		printf("vo_vesa: Can't swap non byte aligned data\n");
		vesa_term();
		exit(-1);
	case 16: *(image + offset) = ByteSwap16(*(image + offset));
		 byte_len = 2; break;
	case 24: ch = *(image+offset);
		 *(image+offset) = *(image+offset+3);
                 *(image+offset+3) = ch;
		 byte_len = 3; break;
	case 32: *(image + offset) = ByteSwap32(*(image + offset));
		 byte_len = 4; break;
       }
       __vbeCopyBlock(offset,image,byte_len);
       size   -= byte_len;
       image  += byte_len;
       offset += byte_len;
   }
}

/*
  Copies frame to video memory. Data should be in the same format as video
