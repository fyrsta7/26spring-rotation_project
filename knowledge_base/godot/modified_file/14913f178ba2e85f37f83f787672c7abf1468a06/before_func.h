	static Vector<int> triangulate_delaunay(const Vector<Vector2> &p_points) {
		Vector<Delaunay2D::Triangle> tr = Delaunay2D::triangulate(p_points);
		Vector<int> triangles;

		for (int i = 0; i < tr.size(); i++) {
			triangles.push_back(tr[i].points[0]);
			triangles.push_back(tr[i].points[1]);
			triangles.push_back(tr[i].points[2]);
		}
		return triangles;
	}
