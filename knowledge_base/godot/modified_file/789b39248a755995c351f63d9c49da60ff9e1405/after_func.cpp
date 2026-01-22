void XRPositionalTracker::set_pose(const StringName &p_action_name, const Transform3D &p_transform, const Vector3 &p_linear_velocity, const Vector3 &p_angular_velocity, const XRPose::TrackingConfidence p_tracking_confidence) {
	Ref<XRPose> new_pose;

	if (poses.has(p_action_name)) {
		new_pose = poses[p_action_name];
	} else {
		new_pose.instantiate();
		poses[p_action_name] = new_pose;
	}

	new_pose->set_name(p_action_name);
	new_pose->set_has_tracking_data(true);
	new_pose->set_transform(p_transform);
	new_pose->set_linear_velocity(p_linear_velocity);
	new_pose->set_angular_velocity(p_angular_velocity);
	new_pose->set_tracking_confidence(p_tracking_confidence);

	emit_signal(SNAME("pose_changed"), new_pose);

	// TODO discuss whether we also want to create and emit an InputEventXRPose event
}
