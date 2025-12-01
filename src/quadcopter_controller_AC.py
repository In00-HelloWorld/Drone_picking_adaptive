import torch
import numpy as np
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.utils.geom import quat_to_xyz

def constrain(v, lo, hi):
    return max(lo, min(v, hi))

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
    
class VelocityMRAC:    
    def __init__(self, kp, ki, kd, gamma: float, nominal_mass: float, device: str = 'cpu'):
        self.pid = PIDController(kp, ki, kd)
        self.gamma = gamma
        # initialize θ ≈ 1/mass
        self.theta = torch.tensor([nominal_mass], dtype=torch.float32)
        self.nominal_mass = nominal_mass

    def update(self, v: float, v_ref: float, e_p: float, dt: float, a: float, a_des: float) -> float:
        # velocity error
        e_v = v_ref - v
        # 1) nominal PID acceleration
        u_nom = self.pid.update(e_p, dt)
        # 2) adaptive augmentation
        # phi = torch.tensor([e_v], dtype=torch.float32, device=self.theta.device)
        u_adapt = ((a_des ) * self.theta).item()
        # 3) adapt θ via MIT rule: θ̇ = -γ * φ * (v - v_ref)
        self.theta -= (self.gamma * (a_des )* (e_p + e_v*0.3)) * dt
        self.theta = constrain(self.theta, 0.0001, 100)
        # self.theta += (self.gamma * (u_nom - a_cur)* u_nom) * dt
        # self.theta = constrain(self.theta, 0, 1000);
        print(self.theta)
        # print(u_adapt / (a_cur + 9.81))
        # print(u_nom - a_cur)
        # print(e_v)
        # total acceleration command
        return u_adapt - u_nom
    
    def new_theta(self, new_mass: float):
        self.theta = torch.tensor([new_mass], dtype=torch.float32)


class DronePIDController:
    def __init__(self, drone: DroneEntity, dt, base_rpm, pid_params, gamma, gamma_a, nominal_mass):
        self.__pid_pos_x = PIDController(kp=pid_params[0][0], ki=pid_params[0][1], kd=pid_params[0][2])
        self.__pid_pos_y = PIDController(kp=pid_params[1][0], ki=pid_params[1][1], kd=pid_params[1][2])
        self.__pid_pos_z = PIDController(kp=pid_params[2][0], ki=pid_params[2][1], kd=pid_params[2][2])

        self.__pid_vel_x = PIDController(kp=pid_params[3][0], ki=pid_params[3][1], kd=pid_params[3][2])
        self.__pid_vel_y = PIDController(kp=pid_params[4][0], ki=pid_params[4][1], kd=pid_params[4][2])
        self.__pid_vel_z = PIDController(kp=pid_params[5][0], ki=pid_params[5][1], kd=pid_params[5][2])

        self.__pid_att_roll = PIDController(kp=pid_params[6][0], ki=pid_params[6][1], kd=pid_params[6][2])
        self.__pid_att_pitch = PIDController(kp=pid_params[7][0], ki=pid_params[7][1], kd=pid_params[7][2])
        self.__pid_att_yaw = PIDController(kp=pid_params[8][0], ki=pid_params[8][1], kd=pid_params[8][2])

        self.thrust = VelocityMRAC(kp=0.8, ki=0, kd=0, gamma = gamma, nominal_mass = nominal_mass)

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm

    def change_base_rpm(self, base_rpm):
        self.__base_rpm = base_rpm

    def change_theta(self, new_mass):
        self.thrust.new_theta(new_mass)

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()
    
    def __get_drone_acc(self) -> torch.Tensor:
        return self.drone.get_links_acc(0).squeeze(-2)

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        return quat_to_xyz(quat, rpy=True, degrees=True)
    
    def __get_drone_ang_acc(self) -> torch.Tensor:
        return self.drone.get_links_acc(0).squeeze(-2)

    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel + y_vel)
        M2 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel + y_vel)
        M3 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel - y_vel)
        M4 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel - y_vel)
        return torch.Tensor([M1, M2, M3, M4])

    def update(self, target) -> np.ndarray:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_acc = self.__get_drone_acc()
        curr_att = self.__get_drone_att()
        curr_ang_acc = self.__get_drone_ang_acc()

        err_pos_x = target[0] - curr_pos[0]
        err_pos_y = target[1] - curr_pos[1]
        err_pos_z = target[2] - curr_pos[2]

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        error_vel_x = vel_des_x - curr_vel[0]
        error_vel_y = vel_des_y - curr_vel[1]
        error_vel_z = vel_des_z - curr_vel[2]

        x_vel_del = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_vel_del = self.__pid_vel_y.update(error_vel_y, self.__dt)
        z_vel_des = self.__pid_vel_z.update(error_vel_z, self.__dt)

        thrust_des = self.thrust.update(curr_vel[2], vel_des_z, err_pos_z, self.__dt, curr_acc[2], z_vel_des)


        err_roll = 0.0 - curr_att[0]
        err_pitch = 0.0 - curr_att[1]
        err_yaw = 0.0 - curr_att[2]

        roll_del = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del = self.__pid_att_yaw.update(err_yaw, self.__dt)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)
        prop_rpms = prop_rpms.cpu()
        prop_rpms - prop_rpms.numpy()

        return prop_rpms
