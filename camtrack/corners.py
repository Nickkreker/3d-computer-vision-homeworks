#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # TODO
    eig_params = dict(blockSize = 7,
                      ksize=3)

    feature_params = dict(blockSize=7,
                          maxCorners=0,
                          minDistance=7,
                          qualityLevel=0.091,
                          useHarrisDetector=False)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))

    image_0_lvl1 = frame_sequence[0]
    image_0_lvl2 = cv2.pyrDown(image_0_lvl1)

    corners_0_lvl1 = cv2.goodFeaturesToTrack(image_0_lvl1, mask=None, **feature_params)

    mask = np.zeros_like(image_0_lvl2, np.uint8)
    mask[:] = 255
    for corn in np.flip(corners_0_lvl1.reshape(-1, 2), axis=1):
        y, x = corn / 2
        cv2.circle(mask, (x, y), 5, 0, -1)

    corners_0_lvl2 = cv2.goodFeaturesToTrack(image_0_lvl2, mask=mask, **feature_params)
    corners_0_lvl2 = corners_0_lvl2 * 2

    #Created corners on two levels
    corners_0 = np.concatenate((corners_0_lvl1.reshape(-1, 2), corners_0_lvl2.reshape(-1, 2)))
    corners = FrameCorners(
        np.arange(len(corners_0_lvl1) + len(corners_0_lvl2)),
        np.concatenate((corners_0_lvl1.reshape(-1, 2), corners_0_lvl2.reshape(-1, 2))),
        np.concatenate((np.ones(len(corners_0_lvl1), dtype=int) * 7, np.ones(len(corners_0_lvl2), dtype=int) * 7 * 2))
    )
    builder.set_corners_at_frame(0, corners)

    image_0 = image_0_lvl1
    
    lvl1_num = len(corners_0_lvl1)
    lvl2_num = len(corners_0_lvl2)
    
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners_1_lvl1, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255), np.uint8(image_1 * 255), corners_0_lvl1, None, **lk_params)
        corners_1_lvl2, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255), np.uint8(image_1 * 255), corners_0_lvl2, None, **lk_params)

        # backtrack
        corners_0_lvl1r, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_1 * 255), np.uint8(image_0 * 255), corners_1_lvl1, None, **lk_params)
        error = abs(corners_0_lvl1 - corners_0_lvl1r).reshape(-1, 2).max(-1)
        corners_1_lvl1 = corners_1_lvl1[error < 1]

        if corners_1_lvl2 is not None:
            corners_0_lvl2r, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_1 * 255), np.uint8(image_0 * 255), corners_1_lvl2, None, **lk_params)
            error = abs(corners_0_lvl2 - corners_0_lvl2r).reshape(-1, 2).max(-1)
            corners_1_lvl2 = corners_1_lvl2[error < 1]
        
        if frame % 5 == 0:
            t1 = len(corners_1_lvl1)

            image_1_lvl2 = cv2.pyrDown(image_1)
            mask = np.zeros_like(image_1, np.uint8)
            mask[:] = 255
            for corn in np.flip(corners_1_lvl1.reshape(-1, 2), axis=1):
                y, x = corn
                cv2.circle(mask, (x, y), 5, 0, -1)
            for corn in np.flip(corners_1_lvl2.reshape(-1, 2), axis=1):
            	y, x = corn
            	cv2.circle(mask, (x, y), 5, 0, -1)

            corners_1_lvl1_new = cv2.goodFeaturesToTrack(image_1, mask=mask, blockSize=7, maxCorners=max(lvl1_num -t1, 5),
                                                     minDistance=7, qualityLevel=0.05, useHarrisDetector=False)
            mask = np.zeros_like(image_1_lvl2, np.uint8)
            mask[:] = 255

            #if corners_1_lvl1 is not None and len(corners_1_lvl1) != 0:
            corners_1_lvl1 = np.concatenate((corners_1_lvl1, corners_1_lvl1_new))
            for corn in np.flip(corners_1_lvl1_new.reshape(-1, 2), axis=1):
                y,x = corn / 2
                cv2.circle(mask, (x, y), 5, 0, -1)
            for corn in np.flip(corners_1_lvl1.reshape(-1, 2), axis=1):
                y,x = corn / 2
                cv2.circle(mask, (x,y), 5, 0, -1)
            for corn in np.flip(corners_1_lvl2.reshape(-1, 2), axis=1):
            	y, x= corn / 2
            	cv2.circle(mask, (x, y), 5, 0, -1)

            t2 = lvl2_num - len(corners_1_lvl2)
            corners_1_lvl2_new = cv2.goodFeaturesToTrack(image_1_lvl2, mask=mask, blockSize=7, maxCorners=max(t2, 1),
                                                         minDistance=7, qualityLevel=0.091, useHarrisDetector=False)
            corners_1_lvl2 = np.concatenate((corners_1_lvl2, corners_1_lvl2_new * 2))
                
    
        corners = FrameCorners(
            np.arange(len(corners_1_lvl1) + len(corners_1_lvl2)),
            np.concatenate((corners_1_lvl1.reshape(-1, 2), corners_1_lvl2.reshape(-1, 2))),
            np.concatenate((np.ones(len(corners_1_lvl1), dtype=int) * 7, np.ones(len(corners_1_lvl2), dtype=int) * 7 * 2)),
        )

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1.copy()
        corners_0_lvl1 = corners_1_lvl1.copy()
        corners_0_lvl2 = corners_1_lvl2.copy()


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
