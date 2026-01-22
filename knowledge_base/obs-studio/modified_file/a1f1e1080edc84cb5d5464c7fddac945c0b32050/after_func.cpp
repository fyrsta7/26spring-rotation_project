void OBSQTDisplay::CreateDisplay()
{
	if (display || !windowHandle()->isExposed())
		return;

	QSize size = GetPixelSize(this);

	gs_init_data info = {};
	info.cx = size.width();
	info.cy = size.height();
	info.format = GS_BGRA;
	info.zsformat = GS_ZS_NONE;

	QTToGSWindow(winId(), info.window);

	display = obs_display_create(&info, backgroundColor);

	emit DisplayCreated(this);
}
