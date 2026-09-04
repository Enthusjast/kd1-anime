from kd1_anime.agents.render_context import renderer_guidance


def test_opengl_guidance_keeps_three_d_scene_available():
    guidance = renderer_guidance("opengl")

    assert "ThreeDScene" in guidance
    assert "ThreeDAxes" in guidance
    assert "Surface" in guidance
    assert "set_camera_orientation" in guidance
    assert "固定视角不等于改用普通 `Scene`" in guidance


def test_opengl_guidance_only_forbids_frame_based_camera_motion():
    guidance = renderer_guidance("opengl")

    assert "禁止 `self.camera.frame`" in guidance
    assert "MovingCameraScene" in guidance
