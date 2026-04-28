from typing import TYPE_CHECKING

import torch
import numpy as np

from genesis.utils.geom import quat_to_xyz

if TYPE_CHECKING:
    from genesis.engine.entities.drone_entity import DroneEntity


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class DronePIDController:
    def __init__(self, drone: "DroneEntity", dt, base_rpm, pid_params):
        self.__pid_pos_x = PIDController(kp=pid_params[0][0], ki=pid_params[0][1], kd=pid_params[0][2])
        self.__pid_pos_y = PIDController(kp=pid_params[1][0], ki=pid_params[1][1], kd=pid_params[1][2])
        self.__pid_pos_z = PIDController(kp=pid_params[2][0], ki=pid_params[2][1], kd=pid_params[2][2])

        self.__pid_vel_x = PIDController(kp=pid_params[3][0], ki=pid_params[3][1], kd=pid_params[3][2])
        self.__pid_vel_y = PIDController(kp=pid_params[4][0], ki=pid_params[4][1], kd=pid_params[4][2])
        self.__pid_vel_z = PIDController(kp=pid_params[5][0], ki=pid_params[5][1], kd=pid_params[5][2])

        self.__pid_att_roll = PIDController(kp=pid_params[6][0], ki=pid_params[6][1], kd=pid_params[6][2])
        self.__pid_att_pitch = PIDController(kp=pid_params[7][0], ki=pid_params[7][1], kd=pid_params[7][2])
        self.__pid_att_yaw = PIDController(kp=pid_params[8][0], ki=pid_params[8][1], kd=pid_params[8][2])

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm

        # Basic adaptive XY disturbance compensation (disabled by default).
        self.__adaptive_xy_enabled = False
        self.__gamma_xy = 0.03  # adaptation gain
        self.__leak_xy = 0.05   # leakage (damping) for stability
        self.__d_hat_xy_max = 3.0
        self.__adapt_ev_deadzone = 0.05
        self.__d_hat_x = 0.0
        self.__d_hat_y = 0.0

        # XY tracking safety limits (prevents lateral control blow-up).
        self.__vel_des_xy_max = 1.0
        self.__xy_mix_limit = 500.0

        # Last-step debug snapshot for offline XY adaptation analysis.
        self.__last_xy_debug = {
            "adaptive_enabled": 0.0,
            "err_pos_x": 0.0,
            "err_pos_y": 0.0,
            "vel_des_x": 0.0,
            "vel_des_y": 0.0,
            "error_vel_x": 0.0,
            "error_vel_y": 0.0,
            "x_vel_pid": 0.0,
            "y_vel_pid": 0.0,
            "d_hat_x": 0.0,
            "d_hat_y": 0.0,
            "x_vel_cmd": 0.0,
            "y_vel_cmd": 0.0,
        }

    def set_adaptive_xy_enabled(self, enabled: bool = True):
        self.__adaptive_xy_enabled = bool(enabled)

    def reset_adaptive_xy(self, x_hat: float = 0.0, y_hat: float = 0.0):
        self.__d_hat_x = float(np.clip(x_hat, -self.__d_hat_xy_max, self.__d_hat_xy_max))
        self.__d_hat_y = float(np.clip(y_hat, -self.__d_hat_xy_max, self.__d_hat_xy_max))

    def set_adaptive_xy_params(self, gamma: float | None = None, leak: float | None = None, d_hat_max: float | None = None):
        if gamma is not None:
            self.__gamma_xy = max(0.0, float(gamma))
        if leak is not None:
            self.__leak_xy = max(0.0, float(leak))
        if d_hat_max is not None:
            self.__d_hat_xy_max = max(0.0, float(d_hat_max))
            self.reset_adaptive_xy(self.__d_hat_x, self.__d_hat_y)

    def set_adaptive_xy_deadzone(self, ev_deadzone: float):
        self.__adapt_ev_deadzone = max(0.0, float(ev_deadzone))

    def set_xy_tracking_limits(self, vel_des_xy_max: float | None = None, xy_mix_limit: float | None = None):
        if vel_des_xy_max is not None:
            self.__vel_des_xy_max = max(0.0, float(vel_des_xy_max))
        if xy_mix_limit is not None:
            self.__xy_mix_limit = max(0.0, float(xy_mix_limit))

    def reset_pid_states(self):
        self.__pid_pos_x.reset()
        self.__pid_pos_y.reset()
        self.__pid_pos_z.reset()
        self.__pid_vel_x.reset()
        self.__pid_vel_y.reset()
        self.__pid_vel_z.reset()
        self.__pid_att_roll.reset()
        self.__pid_att_pitch.reset()
        self.__pid_att_yaw.reset()

    def get_adaptive_xy_debug(self) -> dict:
        return dict(self.__last_xy_debug)

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        return quat_to_xyz(quat, rpy=True, degrees=True)

    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        M2 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        M3 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        M4 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        return torch.Tensor([M1, M2, M3, M4])

    def update(self, target) -> np.ndarray:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        def _f(v):
            if torch.is_tensor(v):
                return float(v.detach().cpu().item())
            return float(v)

        err_pos_x = float(target[0]) - _f(curr_pos[0])
        err_pos_y = float(target[1]) - _f(curr_pos[1])
        err_pos_z = float(target[2]) - _f(curr_pos[2])

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        vel_des_x = float(np.clip(vel_des_x, -self.__vel_des_xy_max, self.__vel_des_xy_max))
        vel_des_y = float(np.clip(vel_des_y, -self.__vel_des_xy_max, self.__vel_des_xy_max))

        error_vel_x = vel_des_x - _f(curr_vel[0])
        error_vel_y = vel_des_y - _f(curr_vel[1])
        error_vel_z = vel_des_z - _f(curr_vel[2])

        x_vel_del = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_vel_del = self.__pid_vel_y.update(error_vel_y, self.__dt)
        thrust_des = self.__pid_vel_z.update(error_vel_z, self.__dt)

        x_vel_pid = float(x_vel_del)
        y_vel_pid = float(y_vel_del)

        if self.__adaptive_xy_enabled:
            # Use only velocity error for adaptation, with deadzone and leakage
            evx = float(error_vel_x)
            evy = float(error_vel_y)

            if abs(evx) < self.__adapt_ev_deadzone:
                evx = 0.0
            if abs(evy) < self.__adapt_ev_deadzone:
                evy = 0.0

            # Adaptive law with leakage to prevent drift after long transients.
            self.__d_hat_x += self.__dt * (self.__gamma_xy * evx - self.__leak_xy * self.__d_hat_x)
            self.__d_hat_y += self.__dt * (self.__gamma_xy * evy - self.__leak_xy * self.__d_hat_y)
            print(self.__d_hat_x, self.__d_hat_y)
            self.__d_hat_x = float(np.clip(self.__d_hat_x, -self.__d_hat_xy_max, self.__d_hat_xy_max))
            self.__d_hat_y = float(np.clip(self.__d_hat_y, -self.__d_hat_xy_max, self.__d_hat_xy_max))

            x_vel_del = x_vel_del - self.__d_hat_x
            y_vel_del = y_vel_del - self.__d_hat_y

        x_vel_del = float(np.clip(x_vel_del, -self.__xy_mix_limit, self.__xy_mix_limit))
        y_vel_del = float(np.clip(y_vel_del, -self.__xy_mix_limit, self.__xy_mix_limit))

        self.__last_xy_debug = {
            "adaptive_enabled": 1.0 if self.__adaptive_xy_enabled else 0.0,
            "err_pos_x": float(err_pos_x),
            "err_pos_y": float(err_pos_y),
            "vel_des_x": float(vel_des_x),
            "vel_des_y": float(vel_des_y),
            "error_vel_x": float(error_vel_x),
            "error_vel_y": float(error_vel_y),
            "x_vel_pid": float(x_vel_pid),
            "y_vel_pid": float(y_vel_pid),
            "d_hat_x": float(self.__d_hat_x),
            "d_hat_y": float(self.__d_hat_y),
            "x_vel_cmd": float(x_vel_del),
            "y_vel_cmd": float(y_vel_del),
        }

        err_roll = 0.0 - _f(curr_att[0])
        err_pitch = 0.0 - _f(curr_att[1])
        err_yaw = 0.0 - _f(curr_att[2])

        roll_del = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del = self.__pid_att_yaw.update(err_yaw, self.__dt)

        # --- Convert XY velocity commands from world frame to body frame using yaw ---
        # This fixes command authority loss after attitude/payload changes.
        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)

        """ yaw_rad = float(curr_att[2]) * np.pi / 180.0  # Convert yaw from degrees to radians
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        # Rotation matrix: [x_body] = [cos  sin] [x_world]
        #                  [y_body]   [-sin cos] [y_world]
        x_vel_body = cos_yaw * x_vel_del + sin_yaw * y_vel_del
        y_vel_body = -sin_yaw * x_vel_del + cos_yaw * y_vel_del

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_body, y_vel_body)"""

        return prop_rpms.cpu().numpy()