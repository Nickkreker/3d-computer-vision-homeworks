#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'filter_frame_corners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks',
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli, filter_frame_corners


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))

def calc_corners(image, max_level=2, old_corners=None):
	feature_params = dict(blockSize=7,
                          maxCorners=0,
                          minDistance=7,
                          qualityLevel=0.091,
                          useHarrisDetector=False)

	levels = [frame_sequence[0]]
    points = np.empty((0, 1, 2), dtype=np.uint8)
    sizes = np.empty((0), dtype=np.uint8)
    ids = np.empty((0), dtype=np.uint8)

    if old_corners is not None:
    	last_id = (prev_points.ids.ravel()[-1] + 1).astype(int)
    else:
    	last_id = 0

    for i in range(max_level):
        levels.append(cv2.pyrDown(levels[-1]))

    for counter, level in enumerate(levels):
        scale = 2 ** counter


        mask = np.zeros_like(level, np.uint8)
        mask[:] = 255
        if old_corners is not None:
            points_on_level = old_corners.points[(old_corners.sizes == 7 * scale).reshape(-1)]
            for point in points_on_level:
                x, y = point / scale
                cv2.circle(mask, (x, y), 7 * scale, 0, -1)
            
        new_points = cv2.goodFeaturesToTrack(level, mask=mask, **feature_params)

        if new_points is not None:
            points = np.append(points, new_points * scale, axis=0)
            sizes = np.append(sizes, (np.ones(len(new_points), dtype=int) * 7 * scale), axis=0)
    ids = np.arange(last_id, last_id + len(points))
    return ids, points, sizes


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # TODO
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))

    # Create corners on two levels
    image_0 = frame_sequence[0]
    ids, points, sizes = calc_corners(image_0)
    corners = FrameCorners(
        ids,
        points,
        sizes
    )

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        next_points, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255), np.uint8(image_1 * 255),
                                                        corners.points, None, **lk_params)

        # backtrack
        # corners_0_lvl1r, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_1 * 255), np.uint8(image_0 * 255), corners_1_lvl1, None, **lk_params)
        # error = abs(corners_0_lvl1 - corners_0_lvl1r).reshape(-1, 2).max(-1)
        # corners_1_lvl1 = corners_1_lvl1[error < 1]
        tracked_corners = filter_frame_corners(corners, st.reshape(-1) == 1)
        builder.set_corners_at_frame(frame - 1, tracked_corners)
    
        prev_points = FrameCorners(
            tracked_corners.ids,
            next_points[st.reshape(-1) == 1],
            tracked_corners.sizes
        )

        ids, points, sizes = calc_corners(image_1, old_corners=prev_points)
        ids = np.append(prev_points.ids.reshape(-1), ids, axis=0)
        points = np.append(prev_points.points.reshape(-1,2), points.reshape(-1,2), axis=0)
        sizes = np.append(prev_points.sizes.reshape(-1), sizes, axis=0)
    
        corners = FrameCorners(
        	ids,
        	points,
        	sizes
        )

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
