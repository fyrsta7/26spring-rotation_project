                                 &ptr)) {
        ITaskbarList3 *taskbl = ptr;
        taskbl->lpVtbl->HrInit(taskbl);

        HWND hroot = GetAncestor(sys->hvideownd, GA_ROOT);
        RECT video;
        if (show) {
            GetWindowRect(sys->hparent, &video);
            POINT client = {video.left, video.top};
            if (ScreenToClient(hroot, &client))
            {
                unsigned int width = RECTWidth(video);
                unsigned int height = RECTHeight(video);
                video.left = client.x;
                video.top = client.y;
                video.right = video.left + width;
                video.bottom = video.top + height;
            }
        }
        HRESULT hr;
        hr = taskbl->lpVtbl->SetThumbnailClip(taskbl, hroot,
                                                 show ? &video : NULL);
        if ( hr != S_OK )
            msg_Err(obj, "SetThumbNailClip failed: 0x%0lx", hr);

        taskbl->lpVtbl->Release(taskbl);
    }
    CoUninitialize();
}
#endif /* !VLC_WINSTORE_APP */

int CommonControl(vlc_object_t *obj, display_win32_area_t *area, vout_display_sys_win32_t *sys, int query, va_list args)
{
