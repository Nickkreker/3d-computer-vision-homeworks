#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple
import cv2

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    Correspondences,
    rodrigues_and_translation_to_view_mat3x4
)

def untracked_frames(view_mats):
    frames = []
    for frame, view_mat in enumerate(view_mats):
        if view_mat is None:
            frames.append(frame)
    return frames

def calc_camera_pose(frame, corner_storage, cloud, intrinsic_mat):
    corners = corner_storage[frame]
    points3d = []
    points2d = []

    for id, point2d in zip(corners.ids.flatten(), corners.points):
        if id in cloud.keys():
            points2d.append(point2d)
            points3d.append(cloud[id])
    points3d = np.array(points3d, dtype=np.float64)
    points2d = np.array(points2d, dtype=np.float64)

    if len(points2d) < 4:
        return None

    succeeded, r_vec, t_vec, inliers = cv2.solvePnPRansac(objectPoints=points3d,
                                                          imagePoints=points2d,
                                                          cameraMatrix=intrinsic_mat,
                                                          distCoeffs=None,
                                                          reprojectionError=8.0,
                                                          confidence=0.999,
                                                          flags=cv2.SOLVEPNP_EPNP)
    if succeeded == False:
        return None

    inliers = np.asarray(inliers, dtype=np.int).flatten()
    points3d = np.array(points3d)
    points2d = np.array(points2d)
    
    points3d = points3d[inliers]
    points2d = points2d[inliers]

    _, rvec, tvec = cv2.solvePnP(objectPoints=points3d[inliers],
                                 imagePoints=points2d[inliers],
                                 cameraMatrix=intrinsic_mat,
                                 distCoeffs=None,
                                 useExtrinsicGuess=True,
                                 rvec=r_vec,
                                 tvec=t_vec,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec, len(inliers)


def add_new_point(corner, frames_of_corner, view_mats, tvecs, corner_storage, cloud, cur_corners_occurencies, intrinsic_mat):
    frames = []
    for frame in frames_of_corner[corner]:
        if view_mats[frame[0]] is not None:
            frames.append(frame)

    if len(frames) < 2:
        return

    max_dist = 0
    best_frames = [None, None]
    best_ids = [None, None]

    for frame_1 in frames:
        for frame_2 in frames:
            if frame_1 == frame_2:
                continue
            tvec_1 = tvecs[frame_1[0]]
            tvec_2 = tvecs[frame_2[0]]

            d = np.linalg.norm(tvec_1 - tvec_2)
            if max_dist < d:
                max_dist = d
                best_frames = [frame_1[0], frame_2[0]]
                best_ids = [frame_1[1], frame_2[1]]

    if max_dist > 0.01:
        ids = np.array([corner])
        points1 = corner_storage[best_frames[0]].points[np.array([best_ids[0]])]
        points2 = corner_storage[best_frames[1]].points[np.array([best_ids[1]])]
        correspondences = Correspondences(ids, points1, points2)
        points3d, ids, _ = triangulate_correspondences(correspondences,
                                                       view_mats[best_frames[0]],
                                                       view_mats[best_frames[1]],
                                                       intrinsic_mat,
                                                       parameters=TriangulationParameters(2, 1e-3, 1e-4))
        if len(points3d) > 0:
            cloud[ids[0]] = points3d[0]
            cur_corners_occurencies.pop(ids[0], None)

def calc_known_views(instrinsic_mat, indent=5, min_points=1500):
    num_frames = len(corner_storage)
    known_view_1 = (None, None)
    known_view_2 = (None, None)
    num_points = -1

    for frame_1 in range(num_frames):
        for frame_2 in range(frame_1 + indent, num_frames):
            corrs = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
            points_1 = corrs.points_1
            points_2 = corrs.points_2
            
            H, mask_h = cv2.findHomography(points_1, points_2, method=cv2.RANSAC)
            mask_h = mask_h.reshape(-1)
            
            E, mask_e = cv2.findEssentialMat(points_1, points_2, method=cv2.RANSAC, cameraMatrix=intrinsic_mat)
            mask_e = mask_e.reshape(-1)
            
            if mask_h.sum() / mask_e.sum() > 0.5:
                continue
            
            corrs = Correspondences(corrs.ids[mask], points_1[mask], points_2[mask])


            R1, R2, t = cv2.decomposeEssentialMat(E)

            for poss_pose in [Pose(R1.T, R1.T@t), Pose(R1.T, R1.T@(-t), Pose(R2.T, R2.T@t), Pose(R2.T, R2.T@(-t))]:
                points3d, _, _ = triangulate_correspondences(corrs, eye3x4, pose_to_view_mat3x4(poss_pose), intrinsic_mat, triang_params)
                if len(points3d) > num_points:
                    num_points = len(points3d)
                    known_view_1 = (frame_1, view_mat3x4_to_pose(eye3x4))
                    known_view_2 = (frame_2, poss_pose)
    return known_view_1, known_view_2

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    num_frames_retr = 10
    triang_params = TriangulationParameters(max_reprojection_error=8.0,
                                            min_triangulation_angle_deg=1.0,
                                            min_depth=.1)

    frame_count = len(corner_storage)
    view_mats = [None] * frame_count
    tvecs = [None] * frame_count

    frame_1 = known_view_1[0]
    frame_2 = known_view_2[0]

    tvecs[frame_1] = known_view_1[1].t_vec
    tvecs[frame_2] = known_view_2[1].t_vec

    view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat_2 = pose_to_view_mat3x4(known_view_2[1])

    view_mats[frame_1] = view_mat_1
    view_mats[frame_2] = view_mat_2

    cloud = {}

    correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
    points3d, ids, _ = triangulate_correspondences(correspondences,
                                                   view_mat_1,
                                                   view_mat_2,
                                                   intrinsic_mat,
                                                   triang_params)
    for point3d, id in zip(points3d, ids):
        cloud[id] = point3d

    current_corners_occurences = {}
    frames_of_corner = {}

    for i, corners in enumerate(corner_storage):
        for id_in_list, j in enumerate(corners.ids.flatten()):
            current_corners_occurences[j] = 0
            if j not in frames_of_corner.keys():
                frames_of_corner[j] = [[i, id_in_list]]
            else:
                frames_of_corner[j].append([i, id_in_list])

    while len(untracked_frames(view_mats)) > 0:
        untr_frames = untracked_frames(view_mats)

        max_num_inl = -1
        best_frame = -1
        best_rvec = None
        best_tvec = None

        for frame in untr_frames:
            rvec, tvec, num_inl = calc_camera_pose(frame, corner_storage, cloud, intrinsic_mat)
            if rvec is not None:
                if num_inl > max_num_inl:
                    best_frame = frame
                    max_num_inl = num_inl
                    best_rvec = rvec
                    best_tvec = tvec

        if max_num_inl == -1:
            break

        corners_to_add = []
        for id in corner_storage[best_frame].ids.flatten():
            if id in current_corners_occurences.keys():
                current_corners_occurences[id] += 1
                if current_corners_occurences[id] >= 2:
                    corners_to_add.append(id)

        for corner in corners_to_add:
            add_new_point(corner, frames_of_corner, view_mats, tvecs, corner_storage, cloud, current_corners_occurences, intrinsic_mat)

        print(f"Frame: {best_frame}, Inliers: {max_num_inl}")
        print(f"Cloud size: {len(cloud)}")
        view_mats[best_frame] = rodrigues_and_translation_to_view_mat3x4(best_rvec, best_tvec)
        tvecs[best_frame] = best_tvec

    last_mat = None
    for nframe in range(frame_count):
        if view_mats[i] is None:
            view_mats[i] = last_mat
        else:
            last_mat = view_mats[i]

    ids = []
    points = []

    for id in cloud.keys():
        ids.append(id)
        points.append(cloud[id])

    point_cloud_builder = PointCloudBuilder(np.array(ids, dtype=np.int), np.array(points))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
