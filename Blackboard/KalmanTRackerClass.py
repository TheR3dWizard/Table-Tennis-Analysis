import numpy as np
from newprint import NewPrint

class KalmanTracker:
    """Simplified Kalman filter for segment-based ball tracking."""

    def __init__(self, dt=1.0, process_noise=0.5, measurement_noise=3.0):
        self.dt = dt
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 0.95, 0], [0, 0, 0, 0.95]]
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 50
        self.history = []
        self.confidence = []
        print(
            "Initialized Kalman tracker with dt=%.2f, process_noise=%.2f, measurement_noise=%.2f"
            % (dt, process_noise, measurement_noise)
        )
        self.newprint = NewPrint("KalmanTracker").newprint
        self.newprint("Initialized Kalman tracker", skipconsole=True, event="init")

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.newprint("Predicted next state: x=%.2f, y=%.2f, vx=%.2f, vy=%.2f" % tuple(self.state), skipconsole=True, event="predict")
        return self.state[:2].copy()

    def update(self, measurement):
        if measurement is None:
            self.history.append(self.state[:2].copy())
            self.confidence.append(0.3)
            print("No measurement provided, using predicted state")
            return

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.history.append(self.state[:2].copy())
        self.confidence.append(1.0)
        self.newprint("Updated state with measurement: x=%.2f, y=%.2f" % tuple(measurement), skipconsole=True, event="update")
