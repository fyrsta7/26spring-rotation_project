	static Vector<int> triangulate_delaunay(const Vector<Vector2> &p_points) {
		Vector<Delaunay2D::Triangle> tr = Delaunay2D::triangulate(p_points);
		Vector<int> triangles;

		triangles.resize(3 * tr.size());
		int *ptr = triangles.ptrw();
		for (int i = 0; i < tr.size(); i++) {
			*ptr++ = tr[i].points[0];
			*ptr++ = tr[i].points[1];
			*ptr++ = tr[i].points[2];
		}
		return triangles;
	}
