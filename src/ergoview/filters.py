from collections import deque

import numpy as np


class MovingAverageFilter:
    """Filtro de média móvel para suavização de landmarks."""

    def __init__(self, window_size: int = 5) -> None:
        """Inicializa o filtro de média móvel.

        Parameters
        ----------
        window_size : int, optional
            Tamanho da janela móvel, por padrão 5
        """
        self.window_size = window_size
        self.history = {}

    def filter(
        self, landmark_id: int, x: float, y: float, z: float
    ) -> np.array:
        """Aplica o filtro de média móvel.

        Parameters
        ----------
        landmark_id : int
            Identificar único do landmark.
        x : float
            Coordenada x do landmark.
        y : float
            Coordenada y do landmark.
        z : float
            Coordenada z do landmark.

        Returns
        -------
        np.array
            Coordenadas filtradas do landmark.
        """
        if landmark_id not in self.history:
            self.history[landmark_id] = deque(maxlen=self.window_size)
        self.history[landmark_id].append((x, y, z))
        return np.mean(self.history[landmark_id], axis=0)


class ExponentialMovingAverageFilter:
    """Filtro de Média Móvel Exponencial (EMA) para suavização de landmarks."""

    def __init__(self, alpha: float = 0.3) -> None:
        """Inicializa o filtro de Média Móvel Exponencial.

        Parameters
        ----------
        alpha : float, optional
            Fator de suavização, por padrão 0.3
        """
        self.alpha = alpha
        self.history = {}

    def filter(
        self, landmark_id: int, x: float, y: float, z: float
    ) -> np.array:
        """Aplica o filtro EMA para suavização de landmarks.

        Parameters
        ----------
        landmark_id : int
            Identificador único do landmark.
        x : float
            Coordenada x do landmark.
        y : float
            Coordenada y do landmark.
        z : float
            Coordenada z do landmark.

        Returns
        -------
        np.array
            Coordenadas suavizadas do landmark.
        """
        if landmark_id not in self.history:
            self.history[landmark_id] = np.array([x, y, z])
        else:
            self.history[landmark_id] = (
                self.alpha * np.array([x, y, z])
                + (1 - self.alpha) * self.history[landmark_id]
            )
        return self.history[landmark_id]


class ConfidenceFilter:
    """Filtro de confiança para suavização de landmarks."""

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        presence_threshold: float = 0.5,
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self.presence_threshold = presence_threshold

        def is_valid(self, landmark) -> bool:
            """Verifica se o landmark é válido com base na visibilidade e presença.

            Parameters
            ----------
            landmark : Landmark
                Landmark a ser verificado.

            Returns
            -------
            bool
                True se o landmark é válido, False caso contrário.
            """
            return (
                hasattr(landmark, 'visibility')
                and hasattr(landmark, 'presence')
                and landmark.visibility >= self.visibility_threshold
                and landmark.presence >= self.presence_threshold
            )


class StabilityFilter:
    def __init__(self, threshold: float = 0.2, history_size: int = 5) -> None:
        self.threshold = threshold
        self.history_size = history_size
        self.history = {}

    def is_stable(
        self, landmark_id: int, x: float, y: float, z: float
    ) -> bool:
        point = np.array([x, y, z])
        if landmark_id not in self.history:
            self.history[landmark_id] = deque(maxlen=self.history_size)
        self.history[landmark_id].append(point)
        if len(self.history[landmark_id]) < 2:
            return False
        prev = self.history[landmark_id][-2]
        return np.linalg.norm(point - prev) < self.threshold


class OutlierFilter:
    def __init__(self, z_threshold=2.0, history_size=10):
        self.z_threshold = z_threshold
        self.history = {}
        self.history_size = history_size

    def is_outlier(self, landmark_id: int, point: np.array) -> bool:
        if landmark_id not in self.history:
            self.history[landmark_id] = deque(maxlen=self.history_size)
        self.history[landmark_id].append(point)

        if len(self.history[landmark_id]) < 3:
            return False

        data = np.array(self.history[landmark_id])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        z_scores = np.abs((point - mean) / std)
        return np.any(z_scores > self.z_threshold)


class SpatialFilter:
    def __init__(
        self, frame_width: float = 1.0, frame_height: float = 1.0
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

        def is_inside_frame(self, x: float, y: float) -> bool:
            return 0 <= x <= self.frame_width and 0 <= y <= self.frame_height

        def is_pose_valid(self, shoulder_y: float, hip_y: float) -> bool:
            return shoulder_y < hip_y

        def is_spatially_valid(
            self, x: float, y: float, shoulder_y: float, hip_y: float
        ) -> bool:
            return self.is_inside_frame(x, y) and self.is_pose_valid(
                shoulder_y, hip_y
            )


class InterpolationFilter:
    @staticmethod
    def interpolate(p1: np.array, p2: np.array, alpha: float) -> np.array:
        return (1 - alpha) * p1 + alpha * p2