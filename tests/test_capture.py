import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from src.ergoview.capture import (
    Landmark,
    Landmarks,
    PoseProcessor,
    PoseProcessorYOLO,
    detect_person,
)


class TestLandmark(unittest.TestCase):

    def test_landmark_initialization(self):
        lm = Landmark(1.0, 2.0, 3.0)
        self.assertEqual(lm.x, 1.0)
        self.assertEqual(lm.y, 2.0)
        self.assertEqual(lm.z, 3.0)

    def test_landmarks_container(self):
        lm1 = Landmark(1.0, 2.0, 3.0)
        lm2 = Landmark(4.0, 5.0, 6.0)
        landmarks = Landmarks([lm1, lm2], side='Left')
        self.assertEqual(len(landmarks), 2)
        self.assertEqual(landmarks.side, 'left')
        self.assertEqual(landmarks[0], lm1)


class TestPoseProcessor(unittest.TestCase):

    @patch('src.ergoview.capture.initialize_mp_hands')
    @patch('src.ergoview.capture.initialize_mp_pose')
    def test_pose_processor_initialization(self, mock_pose, mock_hands):
        mock_hands.return_value = ('mp_hands', 'hands', 'mp_draw')
        mock_pose.return_value = ('mp_pose', 'pose')

        processor = PoseProcessor(filter_type='ema', alpha=0.5)
        self.assertEqual(processor.filter.alpha, 0.5)
        self.assertIsNotNone(processor.hands)
        self.assertIsNotNone(processor.pose)

    @patch('src.ergoview.capture.initialize_mp_hands')
    @patch('src.ergoview.capture.initialize_mp_pose')
    def test_pose_processor_invalid_filter(self, mock_pose, mock_hands):
        mock_hands.return_value = ('mp_hands', 'hands', 'mp_draw')
        mock_pose.return_value = ('mp_pose', 'pose')

        with self.assertRaises(ValueError):
            PoseProcessor(filter_type='invalid')


class TestPoseProcessorYOLO(unittest.TestCase):

    @patch('src.ergoview.capture.cv2.dnn.blobFromImage')
    def test_detect_person(self, mock_blob_from_image):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_net = MagicMock()
        mock_classes = ["person", "dog"]  # usando duas classes
        mock_output_layers = ["layer1"]

        # simula uma pessoa detectada
        mock_net.forward.return_value = [
            np.array(
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.95,
                        0.1,
                        0.2,
                    ]  # class_id 0, confidence 0.95
                ]
            )
        ]

        yolo = PoseProcessorYOLO(
            pose_processor=MagicMock(),
            net=mock_net,
            classes=mock_classes,
            output_layers=mock_output_layers,
        )

        boxes = yolo.detect_person(frame)
        self.assertEqual(len(boxes), 1)

    @patch('src.ergoview.capture.cv2.dnn.blobFromImage')
    def test_detect_person_none(self, mock_blob_from_image):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_net = MagicMock()
        mock_classes = ["person", "dog"]
        mock_output_layers = ["layer1"]

        # sem nenhuma pessoa detectada
        mock_net.forward.return_value = [
            np.array([[0, 0, 0, 0, 0, 0.3, 0.6, 0.7]])  # confidences < 0.5
        ]

        yolo = PoseProcessorYOLO(
            pose_processor=MagicMock(),
            net=mock_net,
            classes=mock_classes,
            output_layers=mock_output_layers,
        )

        boxes = yolo.detect_person(frame)
        self.assertEqual(len(boxes), 0)


class TestDetectPersonFunction(unittest.TestCase):

    @patch('src.ergoview.capture.cv2.dnn.blobFromImage')
    def test_detect_person_function(self, mock_blob_from_image):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_net = MagicMock()
        mock_classes = ["person", "dog"]
        mock_output_layers = ["layer1"]

        mock_net.forward.return_value = [
            np.array([[0, 0, 0, 0, 0, 0.9, 0.1, 0.2]])  # pessoa detectada
        ]

        boxes = detect_person(
            frame,
            mock_net,
            mock_classes,
            mock_output_layers,
            conf_threshold=0.5,
        )
        self.assertEqual(len(boxes), 1)

    @patch('src.ergoview.capture.cv2.dnn.blobFromImage')
    def test_detect_person_function_no_detection(self, mock_blob_from_image):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_net = MagicMock()
        mock_classes = ["person", "dog"]
        mock_output_layers = ["layer1"]

        mock_net.forward.return_value = [
            np.array([[0, 0, 0, 0, 0, 0.4, 0.7, 0.6]])  # sem pessoa válida
        ]

        boxes = detect_person(
            frame,
            mock_net,
            mock_classes,
            mock_output_layers,
            conf_threshold=0.5,
        )
        self.assertEqual(boxes, [])


if __name__ == '__main__':
    unittest.main()
