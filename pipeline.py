from datetime import datetime

import cv2
import numpy as np


class Pipeline:
    def process(self, frame: np.ndarray) -> np.ndarray:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, now, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        return frame
