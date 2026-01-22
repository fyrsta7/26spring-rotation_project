                type == CV_16UC4 ? (ippiMirror)ippiMirror_16u_C4R :
                type == CV_16SC1 ? (ippiMirror)ippiMirror_16s_C1R :
                type == CV_16SC3 ? (ippiMirror)ippiMirror_16s_C3R :
                type == CV_16SC4 ? (ippiMirror)ippiMirror_16s_C4R :
                type == CV_32SC1 ? (ippiMirror)ippiMirror_32s_C1R :
                type == CV_32SC3 ? (ippiMirror)ippiMirror_32s_C3R :
                type == CV_32SC4 ? (ippiMirror)ippiMirror_32s_C4R :
                type == CV_32FC1 ? (ippiMirror)ippiMirror_32f_C1R :
                type == CV_32FC3 ? (ippiMirror)ippiMirror_32f_C3R :
                type == CV_32FC4 ? (ippiMirror)ippiMirror_32f_C4R : 0;
        }
        IppiAxis axis = flip_mode == 0 ? ippAxsHorizontal :
            flip_mode > 0 ? ippAxsVertical : ippAxsBoth;
        IppiSize roisize = { dst.cols, dst.rows };

        if (ippFunc != 0)
        {
            if (ippFunc(src.ptr(), (int)src.step, dst.ptr(), (int)dst.step, ippiSize(src.cols, src.rows), axis) >= 0)
            {
                CV_IMPL_ADD(CV_IMPL_IPP);
                return;
            }
            setIppErrorStatus();
        }
        else if (ippFuncI != 0)
