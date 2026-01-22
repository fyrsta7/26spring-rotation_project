
		for (Paths::size_type j = 0; j < scaled_path.size(); ++j) {
			polypath.push_back(Point2(
					static_cast<real_t>(scaled_path[j].X) / SCALE_FACTOR,
					static_cast<real_t>(scaled_path[j].Y) / SCALE_FACTOR));
		}
		polypaths.push_back(polypath);
	}
	return polypaths;
}

Vector<Vector<Point2> > Geometry::_polypath_offset(const Vector<Point2> &p_polypath, real_t p_delta, PolyJoinType p_join_type, PolyEndType p_end_type) {

	using namespace ClipperLib;

	JoinType jt = jtSquare;

	switch (p_join_type) {
		case JOIN_SQUARE: jt = jtSquare; break;
		case JOIN_ROUND: jt = jtRound; break;
		case JOIN_MITER: jt = jtMiter; break;
	}

	EndType et = etClosedPolygon;

	switch (p_end_type) {
		case END_POLYGON: et = etClosedPolygon; break;
		case END_JOINED: et = etClosedLine; break;
		case END_BUTT: et = etOpenButt; break;
		case END_SQUARE: et = etOpenSquare; break;
		case END_ROUND: et = etOpenRound; break;
	}
	ClipperOffset co(2.0, 0.25 * SCALE_FACTOR); // Defaults from ClipperOffset.
	Path path;

	// Need to scale points (Clipper's requirement for robust computation).
	for (int i = 0; i != p_polypath.size(); ++i) {
		path << IntPoint(p_polypath[i].x * SCALE_FACTOR, p_polypath[i].y * SCALE_FACTOR);
	}
	co.AddPath(path, jt, et);

	Paths paths;
	co.Execute(paths, p_delta * SCALE_FACTOR); // Inflate/deflate.

	// Have to scale points down now.
	Vector<Vector<Point2> > polypaths;

	for (Paths::size_type i = 0; i < paths.size(); ++i) {
		Vector<Vector2> polypath;

