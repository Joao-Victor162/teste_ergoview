from typing import List, Tuple

import cv2
import numpy as np

from configure import (
    draw_arm_angles,
    draw_hand_landmarks,
    draw_pose_without_hands,
    initialize_mp_hands,
    initialize_mp_pose,
)
from filters import (
    ConfidenceFilter,
    ExponentialMovingAverageFilter,
    InterpolationFilter,
    MovingAverageFilter,
    OutlierFilter,
    SpatialFilter,
    StabilityFilter,
)
from utils import calculate_angle


class Landmark:
    """Representa um ponto de referência (landmark) do corpo"""

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        """Inicializa um ponto de referência

        Parameters
        ----------
        x : float
            Coordenada x
        y : float
            Coordenada y
        z : float, optional
            Coordenada z. Padrão é 0.0
        """
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f'Landmark(x={self.x}, y={self.y}, z={self.z})'


class Landmarks:
    """Representa um conjunto de landmarks."""

    def __init__(self, points: List[Landmark], side: str = 'None') -> None:
        """Inicializa o conjunto de landmarks.

        Parameters
        ----------
        points : List[Landmark]
            Lista de pontos(landmarks) da mão.
        side : str, optional
            Indica o lado da mão , by default 'None'
        """
        self.points = points
        self.side = side.lower()

    def __getitem__(self, index: int) -> Landmark:
        return self.points[index]

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __repr__(self) -> str:
        return f'Landmarks(side={self.side}, points={self.points})'


class PoseProcessor:
    """Processador de quadros (frames) para detecção de pose e mãos.
    Aplica filtros de suavização e ajusta a conexão entre pose e mãos.
    """

    def __init__(
        self,
        filter_type: str = 'ema',
        alpha: float = 0.3,
        window_size: int = 5,
    ) -> None:
        """Inicializa o processador.

        Parameters
        ----------
        filter_type : str, optional
            Tipo de filtro a ser utilizado: 'ema' (exponencial) ou 'ma' (média móvel). Padrão é 'ema'
        alpha : float, optional
            Fator de suavização do EMA. Por padrão é  0.3.
        window_size : int, optional
            Tamanho da janela para a média móvel. Por padrão é 5.

        Raises
        ------
        ValueError
            Se o tipo de filtro for inválido.
        """
        self.mp_hands, self.hands, self.mp_draw = initialize_mp_hands()
        self.mp_pose, self.pose = initialize_mp_pose()

        self.last_pose_landmarks = None

        # Configura os filtros de suavização
        if filter_type == 'ema':
            self.filter = ExponentialMovingAverageFilter(alpha=alpha)
        elif filter_type == 'ma':
            self.filter = MovingAverageFilter(window_size=window_size)
        else:
            raise ValueError(f'Fitro inválido: {filter_type}')

        self.conf_filter = ConfidenceFilter()
        self.stab_filter = StabilityFilter()
        self.outlier_filter = OutlierFilter()
        self.spatial_filter = SpatialFilter()
        self.interpolation_filter = InterpolationFilter()

    def filter_pose_landmarks(
        self, pose_landmarks_raw, frame_shape
    ) -> List[Landmark]:
        h, w = frame_shape[:2]
        filtered_pose_landmarks = []

        for idx, lm in enumerate(pose_landmarks_raw.landmark):
            if not self.conf_filter.is_valid(lm):
                continue
            x, y, z = lm.x * w, lm.y * h, lm.z * w
            point = np.array([x, y, z])
            if not self.stab_filter.is_stable(idx, x, y, z):
                continue
            if self.outlier_filter.is_outlier(idx, point):
                continue
            if idx in [11, 12]:
                hip_idx = 23 if idx == 11 else 24
                if hip_idx < len(pose_landmarks_raw.landmark):
                    hip = pose_landmarks_raw.landmark[hip_idx]
                    if not self.spatial_filter.is_spatially_valid(
                        x=lm.x, y=lm.y, shoulder_y=lm.y, hip_y=hip.y
                    ):
                        continue
            fx, fy, fz = self.filter.filter(idx, x, y, z)
            filtered_pose_landmarks.append(Landmark(fx, fy, fz))

        self.last_pose_landmarks = filtered_pose_landmarks
        return filtered_pose_landmarks

    def filter_hand_landmarks(
        self, hand_landmarks_raw, hand_index: int, frame_shape
    ) -> List[Landmark]:
        h, w = frame_shape[:2]
        filtered_hand_landmarks = []

        for idx, lm in enumerate(hand_landmarks_raw.landmark):
            uid = hand_index * 21 + idx
            if not self.conf_filter.is_valid(lm):
                if self.last_pose_landmarks and uid < len(
                    self.last_pose_landmarks
                ):
                    prev = np.array(
                        [
                            self.last_pose_landmarks[uid].x,
                            self.last_pose_landmarks[uid].y,
                            self.last_pose_landmarks[uid].z,
                        ]
                    )
                    new = np.array([lm.x * w, lm.y * h, lm.z * w])
                    interpolated = self.interp_filter.interpolate(
                        prev, new, alpha=0.5
                    )
                    filtered_hand_landmarks.append(Landmark(*interpolated))
                continue
            x, y, z = lm.x * w, lm.y * h, lm.z * w
            point = np.array([x, y, z])
            if not self.stab_filter.is_stable(uid, x, y, z):
                continue
            if self.outlier_filter.is_outlier(uid, point):
                continue
            fx, fy, fz = self.filter.filter(uid, x, y, z)
            filtered_hand_landmarks.append(Landmark(fx, fy, fz))

        return filtered_hand_landmarks

    def get_arm_angles(
        self, pose_landmarks: List[Landmark]
    ) -> Tuple[float, float]:
        """Calcula os ângulos dos braços com base nos landmarks da pose."""
        if not pose_landmarks or len(pose_landmarks) < 17:
            return ValueError('Landmarks da pose insuficientes.')

        idx_shoulder_r, idx_elbow_r, idx_wrist_r = 12, 14, 16
        idx_shoulder_l, idx_elbow_l, idx_wrist_l = 11, 13, 15

        try:
            shoulder_r, elbow_r, wrist_r = (
                pose_landmarks[idx_shoulder_r],
                pose_landmarks[idx_elbow_r],
                pose_landmarks[idx_wrist_r],
            )
            shoulder_l, elbow_l, wrist_l = (
                pose_landmarks[idx_shoulder_l],
                pose_landmarks[idx_elbow_l],
                pose_landmarks[idx_wrist_l],
            )

            angle_right = calculate_angle(
                (shoulder_r.x, shoulder_r.y),
                (elbow_r.x, elbow_r.y),
                (wrist_r.x, wrist_r.y),
            )
            angle_left = calculate_angle(
                (shoulder_l.x, shoulder_l.y),
                (elbow_l.x, elbow_l.y),
                (wrist_l.x, wrist_l.y),
            )

            return angle_right, angle_left

        except Exception as e:
            raise RuntimeError(f'Erro ao calcular ângulos dos braços: {e}')


class PoseProcessorYOLO:
    def __init__(
        self,
        pose_processor: PoseProcessor,
        net,
        classes,
        output_layers,
        draw_pose_without_hands=draw_pose_without_hands,
        draw_arm_angles=draw_arm_angles,
        draw_hand_landmarks=draw_hand_landmarks,
    ):
        self.pose_processor = pose_processor
        self.net = net
        self.classes = classes
        self.output_layers = output_layers
        self.draw_pose_without_hands = draw_pose_without_hands
        self.draw_arm_angles = draw_arm_angles
        self.draw_hand_landmarks = draw_hand_landmarks

    def detect_person(self, frame, conf_threshold=0.5):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if (
                    confidence > conf_threshold
                    and class_id < len(self.classes)
                    and self.classes[class_id] == "person"
                ):
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = max(0, int(center_x - w / 2))
                    y = max(0, int(center_y - h / 2))
                    boxes.append((x, y, w, h))
        return boxes

    def process_frame(self, frame):
        mp_pose = self.pose_processor.mp_pose
        mp_hands = self.pose_processor.mp_hands
        pose = self.pose_processor.pose
        hands = self.pose_processor.hands
        filter = self.pose_processor.filter
        self.payload_response = []

        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        _ = self.detect_person(frame)

        hand_results = hands.process(image)
        pose_results = pose.process(image)

        hand_landmarks_list = []

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_index, (hand_landmark, handedness) in enumerate(
                zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                )
            ):
                filtered_hand_landmarks = []
                for idx, lm in enumerate(hand_landmark.landmark):
                    x, y, z = lm.x * w, lm.y * h, lm.z * w
                    filtered_x, filtered_y, filtered_z = filter.filter(
                        hand_index * 21 + idx, x, y, z
                    )
                    filtered_hand_landmarks.append(
                        Landmark(filtered_x, filtered_y, filtered_z)
                    )
                hand_side = handedness.classification[0].label
                hand_obj = Landmarks(filtered_hand_landmarks, side=hand_side)
                hand_landmarks_list.append(hand_obj)
            for hand_landmark in hand_results.multi_hand_landmarks:
                self.draw_hand_landmarks(frame, hand_landmark, mp_hands)

        filtered_pose_landmarks = None
        if pose_results.pose_landmarks:
            filtered_pose_landmarks = []
            for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                x, y, z = lm.x * w, lm.y * h, lm.z * w
                filtered_x, filtered_y, filtered_z = filter.filter(
                    idx, x, y, z
                )
                filtered_pose_landmarks.append(
                    Landmark(filtered_x, filtered_y, filtered_z)
                )

            for hand_obj in hand_landmarks_list:
                if (
                    hand_obj.side.lower() == 'left'
                    and len(filtered_pose_landmarks) > 15
                ):
                    hand_obj.points[0] = filtered_pose_landmarks[15]
                elif (
                    hand_obj.side.lower() == 'right'
                    and len(filtered_pose_landmarks) > 16
                ):
                    hand_obj.points[0] = filtered_pose_landmarks[16]

            self.draw_pose_without_hands(
                frame, pose_results.pose_landmarks, mp_pose
            )
            self.payload_response = self.draw_arm_angles(
                frame, self.pose_processor, filtered_pose_landmarks
            )

            #print(f"payload_response{payload_response}")

        return frame, hand_landmarks_list, filtered_pose_landmarks, self.payload_response


def detect_person(frame, net, classes, output_layers, conf_threshold=0.5):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))
                boxes.append((x, y, w, h))
    return boxes
