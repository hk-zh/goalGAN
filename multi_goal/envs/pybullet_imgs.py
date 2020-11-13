cam_height = 30
cam_pos = [7.5, 2.5, cam_height]
target = [*cam_pos[:2], 0]
resolution = (64, 64)
ViewMatrixParams = {
    "cameraEyePosition": cam_pos,
    "cameraTargetPosition": target,
    "cameraUpVector": [0, 1, 0]
}
ProjectionMatrixParams = {
    "fov": 45.0,
    "aspect": 1,
    "nearVal": 0.1,
    "farVal": cam_height+30
}


def get_labyrinth_cam_settings(pybullet):
    return {
        "width": resolution[0],
        "height": resolution[1],
        "viewMatrix": pybullet.computeViewMatrix(**ViewMatrixParams),
        "projectionMatrix": pybullet.computeProjectionMatrixFOV(**ProjectionMatrixParams)
    }
