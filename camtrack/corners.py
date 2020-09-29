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
    image_0_lvl1 = frame_sequence[0]

    #w, h = image_0.shape

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

    image_0_lvl2 = cv2.pyrDown(image_0_lvl1)

    corners_0_lvl1 = cv2.goodFeaturesToTrack(image_0_lvl1, mask=None, **feature_params)
    mask = np.zeros_like(image_0_lvl2, np.uint8)
    mask[:] = 255
    for corn in np.flip(corners_0_lvl1.reshape(-1, 2), axis=1):
        y,x = corn / 2
        cv2.circle(mask, (x, y), 5, 0, -1)

    corners_0_lvl2 = cv2.goodFeaturesToTrack(image_0_lvl2, mask=mask, **feature_params)

#t = np.concatenate((corners_first_lvl, corners_second_lvl), axis=0)
#t = np.concatenate((t, corners_third_lvl), axis=0)
#a = np.concatenate(((np.ones(len(corners_first_lvl), dtype=int) * corner_size),
# np.ones(len(corners_second_lvl), dtype=int) * 4 * corner_size))
#a = np.concatenate((a, np.ones(len(corners_third_lvl))*16*corner_size))

    #Created corners on two levels
    corners_0 = np.concatenate((corners_0_lvl1.reshape(-1, 2), corners_0_lvl2.reshape(-1, 2) * 2))
    corners = FrameCorners(
        np.arange(len(corners_0_lvl1) + len(corners_0_lvl2)),
        np.concatenate((corners_0_lvl1.reshape(-1, 2), corners_0_lvl2.reshape(-1, 2)* 2)),
        np.concatenate((np.ones(len(corners_0_lvl1), dtype=int) * 7, np.ones(len(corners_0_lvl2), dtype=int) * 7 * 2))
    )
    builder.set_corners_at_frame(0, corners)

    image_0 = image_0_lvl1
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners_1, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255), np.uint8(image_1 * 255), corners_0, None, **lk_params)
        
        #backtrack
        corners_0r, st, err = cv2.calcOpticalFlowPyrLK(np.uint8(image_1 * 255), np.uint8(image_0 * 255), corners_1, None, **lk_params)
        error = abs(corners_0 - corners_0r).reshape(-1, 2).max(-1)

        corners_1 = corners_1[error < 1]
        #corners_1 = cv2.goodFeaturesToTrack(image_1, mask=None, **feature_params)
        
        if frame % 5 == 0:
            t1 = len(corners_1)

            image_1_lvl2 = cv2.pyrDown(image_1)
            mask = np.zeros_like(image_1, np.uint8)
            mask[:] = 255
            for corn in np.flip(corners_1.reshape(-1, 2), axis=1):
                y, x = corn
                cv2.circle(mask, (x, y), 5, 0, -1)

            corners_0_lvl1 = cv2.goodFeaturesToTrack(image_1, mask=mask, blockSize=7, maxCorners=max(200-t1, 1), minDistance=7, qualityLevel=0.01, useHarrisDetector=False)
            mask = np.zeros_like(image_0_lvl2, np.uint8)
            mask[:] = 255
            if corners_0_lvl1 is not None:  
                for corn in np.flip(corners_0_lvl1.reshape(-1, 2), axis=1):
                    y,x = corn / 2
                    cv2.circle(mask, (x, y), 5, 0, -1)
                for corn in np.flip(corners_1.reshape(-1, 2), axis=1):
                    y,x = corn / 2
                    cv2.circle(mask, (x,y), 5, 0, -1)

                t2 = cv2.cornerMinEigenVal(image_0_lvl2, **eig_params).max()
                corners_0_lvl2 = cv2.goodFeaturesToTrack(image_0_lvl2, mask=mask, blockSize=7, maxCorners=1, minDistance=7, qualityLevel=0.005/t2, useHarrisDetector=False)
                corners_1 = np.concatenate((corners_1.reshape(-1, 2), corners_0_lvl1.reshape(-1, 2)))
                if corners_0_lvl2 is not None:
                    corners_1 = np.concatenate((corners_1, corners_0_lvl2.reshape(-1, 2) * 2))
    
        corners = FrameCorners(
            np.arange(len(corners_1)),
            corners_1.reshape(-1, 2),
            np.ones(len(corners_1), dtype=int) * 7,
        )

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1.copy()
        corners_0 = corners_1.copy()


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
