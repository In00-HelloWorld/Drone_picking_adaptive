"""
This is a merged version of:
  - Genesis script (with gripper, cameras, LLaVA mass guess, fly_to_point, pick/drop),
  - the core logic (robust CLF filter for u_c + adaptive u_ad + ACP bound for R),
while keeping the controller interface: DronePIDController.update(target) -> (M1,M2,M3,M4) RPMs.

Key design choice (matching your theory):
  - Robust CLF constraint is enforced on u_c (= u_filtered).
  - Adaptive term u_ad is computed separately and added afterward: u = u_c + u_ad.

How we connect to Genesis' physics model (F_i = K_F RPM_i^2):
  - Convert PID RPMs -> collective thrust: T_pid = K_F * sum_i RPM_i^2
  - CLF-QP / half-space projection computes filtered collective thrust T_c (this is your u_c)
  - Adaptive adds thrust T_ad (from theta_hat)
  - Total desired thrust T = T_c + T_ad
  - Preserve PID differential pattern by scaling RPM vector:
        r_new = r_pid * sqrt( (T/(K_F)) / sum(r_pid^2) )
  - Clamp to [min_rpm, max_rpm] and send once per step.
"""

import genesis as gs
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from scipy.spatial.transform import Rotation as R_scipy
from PIL import Image
import torch  # <-- you need this for LLaVA
import control

# Compatibility for environments where torch.compiler.is_compiling is unavailable.
if not hasattr(torch, "compiler"):
    class _TorchCompilerShim:
        @staticmethod
        def is_compiling():
            return False

    torch.compiler = _TorchCompilerShim()
elif not hasattr(torch.compiler, "is_compiling"):
    def _is_compiling_shim():
        return False

    torch.compiler.is_compiling = _is_compiling_shim

from Controller import DronePIDController
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.vis.camera import Camera

from Adaptive import (
    ZCLFParams, ThetaEstimator, ThetaProjBounds,
    ACIQuantileBound, robust_clf_filter
)

from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

from scipy.interpolate import splprep, splev

# -------------------------
# Global constants / RPM limits
# -------------------------
model_path = "lmms-lab/LLaVA-One-Vision-1.5-8B-Instruct"

# base_rpm = (14468.429183500699 + 533 - 10)*np.sqrt(59.82/29)
base_rpm = 14468.429183500699 * np.sqrt(2.5945)
min_rpm = 0.9 * base_rpm
max_rpm = 1.5 * base_rpm

view_0 = (0, 0, 0)
view_1 = (0, 0, 1)

# -------------------------
# LLaVA mass guess (cached)
# -------------------------
_LV_MODEL = None
_LV_PROCESSOR = None

def _load_llava_once():
    global _LV_MODEL, _LV_PROCESSOR
    if _LV_MODEL is None:
        _LV_MODEL = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        _LV_PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

def guess_mass(rgb: np.ndarray) -> float:
    """
    Returns a scalar mass guess (kg).
    NOTE: Your current parsing returns 0; I fixed it to return the first parsed number if available.
    You should calibrate this predictor with CP offline later.
    """
    _load_llava_once()

    pil_img = Image.fromarray(rgb)
    temp_path = "temp_image.png"
    pil_img.save(temp_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": temp_path},
                {"type": "text", "text": "Give me the weight of the object in the sight only with scalar value (in kg)."},
            ],
        }
    ]

    text = _LV_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _LV_PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = _LV_MODEL.generate(**inputs, max_new_tokens=128)
    generated_trim = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = _LV_PROCESSOR.batch_decode(
        generated_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("[LLaVA output]", output_text)

    numbers = re.findall(r'\d+\.?\d*', str(output_text))
    if len(numbers) == 0:
        return 0.0
    return float(numbers[0])

# -------------------------
# Utility functions
# -------------------------
def hover(drone: DroneEntity, rpm=base_rpm):
    drone.set_propellels_rpm([rpm, rpm, rpm, rpm])

def set_geom_rubber_like(geom, friction: float, restitution: float = 0.01):
    if hasattr(geom, "set_friction"):
        geom.set_friction(float(friction))

    if hasattr(geom, "set_restitution"):
        try:
            geom.set_restitution(float(restitution))
        except Exception:
            pass
    elif hasattr(geom, "restitution"):
        try:
            geom.restitution = float(restitution)
        except Exception:
            pass

def safe_render(cam: Camera):
    try:
        return cam.render()
    except ValueError as e:
        if "zero norm quaternions" in str(e):
            print("[WARN] camera render skipped (invalid quaternion).")
            return None
        raise

def clamp_rpm_vec(rpms: np.ndarray) -> np.ndarray:
    return np.clip(rpms, min_rpm, max_rpm)

def compute_cz_from_quat(drone: DroneEntity) -> float:
    """
    cz = world z component of body z-axis.
    Try both quaternion conventions (xyzw, wxyz), choose a plausible cz in [0,1].
    """
    try:
        q = drone.get_quat().cpu().numpy().reshape(-1)
        if q.shape[0] != 4:
            return 1.0

        # try xyzw
        cz_candidates = []
        try:
            Rm = R_scipy.from_quat(q).as_matrix()
            cz_candidates.append(float(Rm[2, 2]))
        except Exception:
            pass

        # try wxyz -> xyzw reorder
        try:
            q2 = np.array([q[1], q[2], q[3], q[0]], dtype=float)
            Rm2 = R_scipy.from_quat(q2).as_matrix()
            cz_candidates.append(float(Rm2[2, 2]))
        except Exception:
            pass

        # pick best plausible candidate
        cz_valid = [c for c in cz_candidates if np.isfinite(c)]
        if len(cz_valid) == 0:
            return 1.0

        # clamp to [-1,1] then map negative to small positive (avoid sign issues)
        cz_valid = [max(-1.0, min(1.0, c)) for c in cz_valid]
        # want cz in (0,1], pick the max positive
        cz_pos = [c for c in cz_valid if c > 0.0]
        cz = max(cz_pos) if len(cz_pos) else abs(max(cz_valid, key=lambda x: abs(x)))
        return float(np.clip(cz, 0.2, 1.0))
    except Exception:
        return 1.0

# -------------------------
# Robust+Adaptive context holder
# -------------------------
class RobustAdaptiveZCtx:
    def __init__(self, dt: float, g: float, KF: float, m0: float,
                 clf: ZCLFParams, theta_est: ThetaEstimator, aci: ACIQuantileBound,
                 T_min: float, T_max: float):
        self.dt = float(dt)
        self.g = float(g)
        self.KF = float(KF)
        self.m0 = float(m0)
        self.clf = clf
        self.theta_est = theta_est
        self.aci = aci
        self.T_min = float(T_min)
        self.T_max = float(T_max)

        self.prev_vz = None  # for accel finite-
        self.prev_tz = None
        self.log = {k: [] for k in
                    ["t","x","y","z","vx","vy","vz","target_x","target_y","target_z",
                     "V","sqrtV","theta_hat","Rbar","score","T_pid","T_c","T_ad","T_total",
                     "cz","margin","scale",
                     "adaptive_enabled","err_pos_x","err_pos_y","vel_des_x","vel_des_y",
                     "error_vel_x","error_vel_y","x_vel_pid","y_vel_pid",
                     "d_hat_x","d_hat_y","x_vel_cmd","y_vel_cmd"]}

def robust_adaptive_step(
    t: float,
    drone: DroneEntity,
    target: tuple[float, float, float],
    rpms_pid: np.ndarray,
    ctx: RobustAdaptiveZCtx,
    controller: DronePIDController | None = None,
):
    """
    One step of:
      - convert rpms_pid -> thrust T_pid
      - CLF filter -> T_c (this is your u_c)
      - adaptive -> T_ad
      - total -> T_total
      - scale rpms_pid to achieve T_total while preserving differential pattern
      - update ACP bound and theta estimator
      - return rpms_new plus debug values
    """
    dt, g, KF, m0 = ctx.dt, ctx.g, ctx.KF, ctx.m0

    pos = drone.get_pos().cpu().numpy().reshape(-1)
    vel = drone.get_vel().cpu().numpy().reshape(-1)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])

    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])

    cz = compute_cz_from_quat(drone)

    # clamp PID rpms first
    rpms_pid = clamp_rpm_vec(np.asarray(rpms_pid, dtype=float).reshape(4,))
    s_pid = float(np.sum(rpms_pid ** 2))
    T_pid = KF * s_pid  # thrust magnitude (N) along body z-axis

    if ctx.prev_tz is None:
        v_d = 0.0
    else:
        v_d = (tz - ctx.prev_tz) / dt
    ctx.prev_tz = tz

    # tracking error for z only
    e_z = z - tz
    e_v = vz - v_d  # assume v_d = 0 per segment
    e = np.array([e_z, e_v], dtype=float)

    # measured z-accel
    if ctx.prev_vz is None:
        zdd_meas = 0.0
    else:
        zdd_meas = (vz - ctx.prev_vz) / dt
    ctx.prev_vz = vz

    # use previous Rbar in CLF constraint (standard online pattern)
    Rbar_prev = float(ctx.aci.Rbar)
    rho_bar_prev = float(ctx.aci.rho_bar)

    # Error dynamics model (design) for CLF:
    # e_dot = [e_v;  e_vdot], with e_vdot approx = (cz/m0) * T_c - g + R  (zdd_d = 0)
    f = np.array([e_v, -g], dtype=float)
    B = np.array([0.0, (cz / m0)], dtype=float)

    # --- CLF filter for u_c on T_c (u_ad NOT inside this CLF) ---
    # Reference input for QP is T_pid.
    # We call robust_clf_filter with uad=0, and interpret returned u as T_c.
    T_c, delta_T, V, Vx = robust_clf_filter(
        e=e, f=f, B=B,
        uc=T_pid, uad=0.0,
        rho_bar=rho_bar_prev,
        clf=ctx.clf,
        u_limits=(ctx.T_min, ctx.T_max),
    )
    # T_c = T_pid
    # delta_T is just "correction" (implementation detail); T_c is your u_c.

    # --- adaptive term (separate) ---
    theta_hat = float(ctx.theta_est.theta_hat)
    # A simple physically meaningful adaptive thrust: add estimated extra weight, compensate tilt
    T_ad = (g * theta_hat) / cz
    T_total = float(np.clip(T_c + T_ad, ctx.T_min, ctx.T_max))
    T_total = T_pid

    # --- Update ACP bound using residual score ---
    m_hat = m0 + theta_hat
    zdd_pred = (cz / max(m0, 1e-6)) * T_total - g - g/m0 * theta_hat  # predicted zdd with adaptive term
    score = zdd_meas - zdd_pred
    Rbar = ctx.aci.update(score)  # used next step
    score = abs(score)   # Vx = [dV/de_z, dV/de_v]
    rho_bar = ctx.aci.update(np.linalg.norm(Vx) * Rbar_prev)
    # --- adaptive law update ---
    # Use PY consistent with a matched weight-like term in e_vdot:
    # choose PY = [0, -g/m0]  (simple, stable proxy)
    PY = np.array([0.0, -g / m0], dtype=float)
    theta_hat_new = ctx.theta_est.step(dt=dt, PY=PY, Vx=Vx)

    # --- map collective thrust command to rotor RPMs ---
    # Use equal offset in squared-RPM space first:
    #   s_i = rpm_i^2,  sum(s_i) = T/KF
    #   s_i_new = s_i + c  (same c for all rotors)
    # This keeps pairwise differences in s_i unchanged, so commanded attitude moments
    # are preserved better than multiplying all RPMs by one scalar.
    if s_pid < 1e-12:
        scale = 1.0
        rpms_new = rpms_pid.copy()
    else:
        s_des = T_total / KF
        s_vec = rpms_pid ** 2
        c = (s_des - s_pid) / 4.0
        s_new = s_vec + c

        if float(np.min(s_new)) > 0.0:
            rpms_new = np.sqrt(s_new)
        else:
            # Fallback to multiplicative scaling if additive shift is infeasible.
            scale_fallback = math.sqrt(max(s_des / s_pid, 0.0))
            rpms_new = rpms_pid * scale_fallback

        # For logging, report effective energy scaling in squared-RPM space.
        s_new_sum = float(np.sum(rpms_new ** 2))
        scale = math.sqrt(max(s_new_sum / s_pid, 0.0))

    rpms_new = clamp_rpm_vec(rpms_new)

    # --- margin for logging (robust CLF constraint on T_c) ---
    # margin = Vx(f + B*T_c) + ||Vx|| Rbar_prev + 2 alpha V   (<=0 desired)
    margin = float(np.dot(Vx, f + B * T_c) + np.linalg.norm(Vx) * Rbar_prev + 2.0 * ctx.clf.alpha * V)

    # log
    ctx.log["t"].append(t)
    ctx.log["x"].append(x); ctx.log["y"].append(y); ctx.log["z"].append(z)
    ctx.log["vx"].append(vx); ctx.log["vy"].append(vy); ctx.log["vz"].append(vz)
    ctx.log["target_x"].append(tx); ctx.log["target_y"].append(ty); ctx.log["target_z"].append(tz)
    ctx.log["V"].append(V)
    ctx.log["sqrtV"].append(math.sqrt(max(V, 0.0)))
    ctx.log["theta_hat"].append(theta_hat_new)
    ctx.log["Rbar"].append(Rbar)
    ctx.log["score"].append(score)
    ctx.log["T_pid"].append(T_pid)
    ctx.log["T_c"].append(T_c)
    ctx.log["T_ad"].append(T_ad)
    ctx.log["T_total"].append(T_total)
    ctx.log["cz"].append(cz)
    ctx.log["margin"].append(margin)
    ctx.log["scale"].append(scale)

    if controller is not None and hasattr(controller, "get_adaptive_xy_debug"):
        dbg = controller.get_adaptive_xy_debug()
        for key in [
            "adaptive_enabled", "err_pos_x", "err_pos_y", "vel_des_x", "vel_des_y",
            "error_vel_x", "error_vel_y", "x_vel_pid", "y_vel_pid",
            "d_hat_x", "d_hat_y", "x_vel_cmd", "y_vel_cmd",
        ]:
            ctx.log[key].append(float(dbg.get(key, 0.0)))
    else:
        for key in [
            "adaptive_enabled", "err_pos_x", "err_pos_y", "vel_des_x", "vel_des_y",
            "error_vel_x", "error_vel_y", "x_vel_pid", "y_vel_pid",
            "d_hat_x", "d_hat_y", "x_vel_cmd", "y_vel_cmd",
        ]:
            ctx.log[key].append(0.0)

    return rpms_new

# -------------------------
# Your pick/drop helpers (kept mostly unchanged)
# -------------------------
def pick_up_box(
    scene: gs.Scene,
    drone: DroneEntity,
    cam: Camera,
    cam2: Camera,
    controller: DronePIDController,
    ctx: RobustAdaptiveZCtx,
    hold_target: tuple[float, float, float],
):
    print("pick up")

    left = drone.get_joint("joint_left_finger")
    right = drone.get_joint("joint_right_finger")
    up = drone.get_joint("joint_up_finger")
    down = drone.get_joint("joint_down_finger")
    gripper_dofs = [left.dof_idx, right.dof_idx, up.dof_idx, down.dof_idx]

    pos0 = drone.get_pos().cpu().numpy().reshape(-1)
    # hold_target = (float(pos0[0]), float(pos0[1]), float(pos0[2]))

    def tracking_hold_step():
        M1, M2, M3, M4 = controller.update(hold_target)
        rpms_pid = np.array([M1, M2, M3, M4], dtype=float)
        t = (len(ctx.log["t"]) * ctx.dt)
        rpms_new = robust_adaptive_step(t, drone, hold_target, rpms_pid, ctx, controller=controller)
        drone.set_propellels_rpm(rpms_new.astype(np.float32))

        scene.step()
        safe_render(cam)
        cam.set_pose(lookat=view_0)

    def get_gripper_dof_pos():
        if not hasattr(drone, "get_dofs_position"):
            return None
        try:
            dof_pos = drone.get_dofs_position(dofs_idx_local=gripper_dofs)
        except TypeError:
            dof_pos = drone.get_dofs_position()
            dof_pos = dof_pos[gripper_dofs]

        if hasattr(dof_pos, "cpu"):
            dof_pos = dof_pos.cpu().numpy()
        dof_pos = np.asarray(dof_pos, dtype=float).reshape(-1)
        if dof_pos.shape[0] < 4:
            return None
        return dof_pos[:4]

    def close_gripper_rate_limited(target_pos: float, max_delta_per_step: float, max_steps: int, assist_force: float | None = None):
        target_vec = np.array([target_pos, target_pos, target_pos, target_pos], dtype=float)
        for _ in range(max_steps):
            curr = get_gripper_dof_pos()

            if curr is None:
                drone.control_dofs_position(target_vec, dofs_idx_local=gripper_dofs)
            else:
                err = target_vec - curr
                if float(np.max(np.abs(err))) < 3e-4:
                    break
                cmd = curr + np.clip(err, -max_delta_per_step, max_delta_per_step)
                drone.control_dofs_position(cmd, dofs_idx_local=gripper_dofs)

            if assist_force is not None:
                drone.control_dofs_force(
                    force=np.array([assist_force, assist_force, assist_force, assist_force], dtype=float),
                    dofs_idx_local=gripper_dofs,
                )
            tracking_hold_step()

    for _ in range(5):
        tracking_hold_step()

    # staged close: slower near-contact to reduce impact impulse on can/drone.
    close_gripper_rate_limited(target_pos=0.0200, max_delta_per_step=0.0012, max_steps=30, assist_force=0.20)
    close_gripper_rate_limited(target_pos=0.0280, max_delta_per_step=0.0008, max_steps=40, assist_force=0.30)
    close_gripper_rate_limited(target_pos=0.0323, max_delta_per_step=0.00035, max_steps=70, assist_force=0.45)

    """    for _ in range(20):
        hover(drone)
        scene.step()
        safe_render(cam)
        cam.set_pose(lookat=view_0)

    for _ in range(20):
        hover(drone, base_rpm * np.sqrt(600 / 420))
        scene.step()
        safe_render(cam)
        cam.set_pose(lookat=view_0)

    drone.control_dofs_force(force=np.array([1.4, 1.4, 1.4, 1.4]), dofs_idx_local=gripper_dofs)"""

    # After contact, lock to a symmetric finger position to reduce off-center squeeze drift.
    try:
        dof_pos = None
        if hasattr(drone, "get_dofs_position"):
            try:
                dof_pos = drone.get_dofs_position(dofs_idx_local=gripper_dofs)
            except TypeError:
                all_pos = drone.get_dofs_position()
                dof_pos = all_pos[gripper_dofs]

        if dof_pos is not None:
            if hasattr(dof_pos, "cpu"):
                dof_pos = dof_pos.cpu().numpy()
            dof_pos = np.asarray(dof_pos, dtype=float).reshape(-1)

            if dof_pos.shape[0] >= 4:
                left_pos, right_pos, up_pos, down_pos = [float(v) for v in dof_pos[:4]]
                lr_mid = 0.5 * (left_pos + right_pos)
                ud_mid = 0.5 * (up_pos + down_pos)
                sym_pos = np.array([lr_mid, lr_mid, ud_mid, ud_mid], dtype=float)
                for _ in range(30):
                    drone.control_dofs_position(sym_pos, dofs_idx_local=gripper_dofs)
                    tracking_hold_step()
                print(f"[gripper] symmetric hold lr={lr_mid:.4f}, ud={ud_mid:.4f}")
    except Exception as e:
        print(f"[gripper] symmetric position hold skipped: {e}")
    
    drone.control_dofs_force(force=np.array([0.7, 0.7, 0.7, 0.7], dtype=float), dofs_idx_local=gripper_dofs)


def drop_box(scene: gs.Scene, drone: DroneEntity, cam: Camera, controller: DronePIDController):
    print("drop")

    left = drone.get_joint("joint_left_finger")
    right = drone.get_joint("joint_right_finger")
    up = drone.get_joint("joint_up_finger")
    down = drone.get_joint("joint_down_finger")
    gripper_dofs = [left.dof_idx, right.dof_idx, up.dof_idx, down.dof_idx]

    drone.control_dofs_position(np.array([0, 0, 0, 0]), dofs_idx_local=gripper_dofs)

    for _ in range(150):
        hover(drone)
        scene.step()
        safe_render(cam)
        cam.set_pose(lookat=view_0)

# -------------------------
# Your fly_to_point, now with robust/adaptive wrapper
# -------------------------
def fly_to_point(
    target,
    controller: DronePIDController,
    scene: gs.Scene,
    cam: Camera,
    Tstep: int,
    Fig: int,
    ctx: RobustAdaptiveZCtx,
):
    drone = controller.drone
    step = 0

    pos = drone.get_pos().cpu().numpy()
    distance = math.dist(pos, target)

    # fixed goal for plot (your original)
    desired_x, desired_y, desired_z = 1.7, 0.3, 0.8
    x_list, y_list, z_list, time_list = [], [], [], []

    while (distance > 0.001) and step < Tstep:
        # baseline PID gives RPMs
        M1, M2, M3, M4 = controller.update(target)
        rpms_pid = np.array([M1, M2, M3, M4], dtype=float)

        # robust/adaptive modifies only collective via scaling
        t = (len(ctx.log["t"]) * ctx.dt)  # global time from logs
        rpms_new = robust_adaptive_step(t, drone, target, rpms_pid, ctx, controller=controller)

        # apply once per step
        drone.set_propellels_rpm(rpms_new.astype(np.float32))

        scene.step()
        safe_render(cam)
        cam.set_pose(lookat=view_0)

        pos = drone.get_pos().cpu().numpy()
        distance = math.dist(pos, target)
        print("[pos]", pos, "dist", distance)
        step += 1

        if Fig == 1:
            x_list.append(pos[0])
            y_list.append(pos[1])
            z_list.append(pos[2])
            time_list.append(step / 100.0)

    if Fig == 1 and len(time_list) > 2:
        plt.figure(figsize=(10, 6))
        plt.plot(time_list, x_list, label="x position (actual)", color="red", linestyle="-")
        plt.hlines(desired_x, time_list[0], time_list[-1], colors="r", linestyles="--", label="x desired")

        plt.plot(time_list, y_list, label="y position (actual)", color="green", linestyle="-")
        plt.hlines(desired_y, time_list[0], time_list[-1], colors="g", linestyles="--", label="y desired")

        plt.plot(time_list, z_list, label="z position (actual)", color="blue", linestyle="-")
        plt.hlines(desired_z, time_list[0], time_list[-1], colors="b", linestyles="--", label="z desired")

        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("Drone Position vs Fixed Target Position")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("drone_fixed_target_plot.png", dpi=300)
        plt.close()

    return float(distance)

def waypoint_step_budget(drone: DroneEntity, target, base: int = 18, gain: float = 120.0, min_steps: int = 18, max_steps: int = 160) -> int:
    pos = drone.get_pos().cpu().numpy().reshape(-1)
    d = float(math.dist(pos, target))
    steps = int(base + gain * d)
    return int(np.clip(steps, min_steps, max_steps))

def stabilize_after_pickup(
    scene: gs.Scene,
    drone: DroneEntity,
    controller: DronePIDController,
    cam: Camera,
    ctx: RobustAdaptiveZCtx,
    hold_target: tuple[float, float, float] | None = None,
    hover_up: float = 0.08,
    max_steps: int = 600,
    speed_tol: float = 0.06,
    stable_steps_required: int = 120,
    min_hover_steps: int = 220,
):
    if hold_target is None:
        pos0 = drone.get_pos().cpu().numpy().reshape(-1)
        hold_target = (float(pos0[0]), float(pos0[1]), float(pos0[2] + hover_up))

    stable_count = 0
    settled = False
    for step_idx in range(max_steps):
        M1, M2, M3, M4 = controller.update(hold_target)
        rpms_pid = np.array([M1, M2, M3, M4], dtype=float)
        t = (len(ctx.log["t"]) * ctx.dt)
        rpms_new = robust_adaptive_step(t, drone, hold_target, rpms_pid, ctx, controller=controller)
        drone.set_propellels_rpm(rpms_new.astype(np.float32))

        scene.step()
        safe_render(cam)
        cam.set_pose(lookat=view_0)

        vel = drone.get_vel().cpu().numpy().reshape(-1)
        speed = float(np.linalg.norm(vel))
        if np.isfinite(speed) and speed < speed_tol:
            stable_count += 1
        else:
            stable_count = 0

        if step_idx >= min_hover_steps and stable_count >= stable_steps_required:
            print(f"[stabilize] settled: speed < {speed_tol} m/s for {stable_steps_required} steps")
            settled = True
            break

    if not settled:
        print("[stabilize] max_steps reached before settling condition.")

    return settled

def lqr_P_for_z(m0: float, cz0: float = 1.0, q1: float = 10.0, q2: float = 2.0, r: float = 1.0):
    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    B = np.array([[0.0],
                  [cz0 / m0]])   # input is thrust (N) -> e_vdot
    Q = np.diag([q1, q2])
    R = np.array([[r]])
    K, P, eigs = control.lqr(A, B, Q, R)   # P is the Riccati solution
    return np.array(P), np.array(K), np.array(eigs)

def check_ctrb_obs(A,B,Q):
    # sufficient-condition checks
    Co = control.ctrb(A,B)
    rank_ctrb = np.linalg.matrix_rank(Co)
    # observability uses sqrt(Q) as output matrix; for diagonal Q, sqrt is easy:
    C = np.linalg.cholesky(Q + 1e-12*np.eye(Q.shape[0]))  # safe
    Ob = control.obsv(A,C)
    rank_obsv = np.linalg.matrix_rank(Ob)
    return rank_ctrb, rank_obsv


def smooth_drone_path(points, num_points=100, s=0.1, dim=3):
    
    data = np.array(points).T
    
    tck, u = splprep(data, s=s, k=dim)

    u_new = np.linspace(0, 1, num_points)
    
    smoothed_data = splev(u_new, tck)
    
    return np.array(smoothed_data).T

# -------------------------
# Main
# -------------------------
def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            world_frame_size=0.8,
        ),
    )

    plane_0 = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    # Optional translucent panel; keep disabled to avoid a floating transparent bound in the scene.
    plane_1 = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane_light.urdf", euler=(0, 90, 45), fixed=True))

    drone = scene.add_entity(
        morph=gs.morphs.Drone(file="urdf/drones/cf2x_gripper3.urdf", pos=(0.5, 0.5, 0.5), scale=2.3)
    )
    # drone = scene.add_entity(...)

    # 1) Current friction
    for link in drone.links:
        if "gripper" in link.name:
            for geom in link.geoms:
                print(link.name, geom.friction)

    # 2) Increasing finger friction
    for link in drone.links:
        if link.name in ["gripper_left", "gripper_right", "gripper_up", "gripper_down"]:
            for geom in link.geoms:
                # geom.set_friction(1.2) 
                set_geom_rubber_like(geom, friction=0.5, restitution=0.2)  # 마찰과 약간의 반발력 추가
    # objects
    """bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=25),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.05,
            euler=(0, 0, 90),
            pos=(-0.4, 0.9, 0.6),
            fixed=True,
        ),
    )"""

    # Arrange props in a visually balanced layout while keeping the pickup corridor near the coke can clear.
    # Duck = scene.add_entity(gs.morphs.URDF(file="urdf/data/duck_vhacd.urdf", scale=1, euler=(90, 0, 20), pos=(1.10, 0.66, 0.06), fixed=False))
    # Teddy = scene.add_entity(gs.morphs.URDF(file="urdf/data/teddy_vhacd.urdf", scale=1, euler=(90, 0, -15), pos=(1.06, 0.58, 0.07), fixed=False))
    # Racecar = scene.add_entity(gs.morphs.URDF(file="urdf/data/racecar/racecar.urdf", scale=0.3, euler=(0, 0, -25), pos=(1.46, 0.82, 0.04), fixed=False))
    # Lego = scene.add_entity(gs.morphs.URDF(file="urdf/data/lego/lego.urdf", scale=1, euler=(0, 0, 10), pos=(0.92, 0.86, 0.05), fixed=False))

    
    # Table = scene.add_entity(gs.morphs.URDF(file="urdf/data/table/table.urdf", scale=0.5, euler=(0, 0, 90), pos=(0.0, 1.8, 0.5), fixed=False))
    # sJenga = scene.add_entity(gs.morphs.URDF(file="urdf/data/jenga/jenga.urdf", scale=1, euler=(0, 0, 0), pos=(1, 0.92, 0.50), fixed=False))
    # Pan = scene.add_entity(gs.morphs.URDF(file="urdf/data/dinnerware/pan_tefal.urdf", scale=0.5, pos=(-0.2, 0.7, 0.6), fixed=False))
    # Plate = scene.add_entity(gs.morphs.URDF(file="urdf/data/dinnerware/plate.urdf", scale=0.5, pos=(-0.4, 0.7, 0.6), fixed=False))
    
    CokeCan = scene.add_entity(gs.morphs.URDF(file="urdf/coke_can/coke_can.urdf", scale=0.5, pos=(1.7, 0.3, 0.05), fixed=False))

    """CokeCan = scene.add_entity(
        gs.morphs.Cylinder(
            height=0.12,          # Can height (Unit: m)
            radius=0.03,          # Can radius (Unit: m)
            pos=(1.7, 0.3, 0.06), # Center coordinate (0.06 = Half of z height = tangent with the floor)
            fixed=False,          # Fix it or not
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.1, 0.1, 1.0) 
        )
    )"""
    

    pid_params = [
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [20.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [10.0, 0.0, 10.0],
        [15.0, 0.0, 10.0],
        [15.0, 0.0, 10.0],
        [2.0, 0.0, 0.2],
    ]

    controller = DronePIDController(drone, dt=0.01, base_rpm=base_rpm, pid_params=pid_params)
    # Conservative XY adaptation tuned for post-pick payload disturbance.
    controller.set_adaptive_xy_params(gamma=0.1, leak=0.20, d_hat_max=2.0)
    controller.set_adaptive_xy_deadzone(0.03)
    controller.set_xy_tracking_limits(vel_des_xy_max=2.0, xy_mix_limit=180.0)
    controller.set_adaptive_xy_enabled(False)

    cam = scene.add_camera(pos=(6, 6, 4), lookat=view_0, GUI=False, res=(1280, 960), fov=30)
    cam2 = scene.add_camera(pos=(2, 2, 1), lookat=view_0, GUI=False, res=(1280, 960), fov=30)

    # --- physics constants for robust/adaptive ---
    g = 9.81
    KF = float(drone.KF)
    dt = 0.01
    # nominal mass estimate from hover relation mg = 4*KF*r^2
    m0 = (4.0 * KF * base_rpm * base_rpm) / g
    print(f"Initial mass estimate m0: {m0:.3f} kg (calibrated from hover)")

    # --- CLF / ACP / theta estimator ---
    # after KF, m0 computed:
    P_lqr, K_lqr, eigs = lqr_P_for_z(m0=m0, cz0=1.0, q1=20.0, q2=5.0, r=0.50)
    clf = ZCLFParams(alpha=0.5, P=P_lqr)
    print("LQR eigs:", eigs)

    # CP interval placeholder (replace later with your CP-calibrated interval)
    theta_bounds = ThetaProjBounds(theta_min=0.0, theta_max=10.0)

    theta_est = ThetaEstimator(Gamma=0.1, bounds=theta_bounds)
    theta_est.reset(0.0)  # temporary init (will be overwritten by guess_mass if you want)

    aci = ACIQuantileBound(alpha_target=0.1, window=400, eta=0.1)
    

    # thrust limits (N) – tune
    T_min, T_max = 0.0, 10.0

    ctx = RobustAdaptiveZCtx(
        dt=dt, g=g, KF=KF, m0=m0,
        clf=clf, theta_est=theta_est, aci=aci,
        T_min=T_min, T_max=T_max
    )

    # build
    scene.build()
    rigid = scene.sim.rigid_solver

    # optional: adjust link masses (your original)
    base_link = drone.get_link("base_link")
    base_link.set_mass(0.05)
    for name in ["gripper_left", "gripper_right", "gripper_up", "gripper_down"]:
        drone.get_link(name).set_mass(0.005)

    # camera attach (your original)
    camera_forward = np.array([1, 0, 0])
    target_dir = np.array([0.7, 0.0, 1])
    rot, _ = R_scipy.align_vectors(np.vstack([target_dir]), np.vstack([camera_forward]))
    offset_T = np.eye(4)
    offset_T[:3, :3] = rot.as_matrix()

    # NOTE: keep cam2 detached during flight to avoid renderer crash when attached link quaternion becomes invalid.

    cam.start_recording()

    points_before = [
        (0.9, 0.3, 0.6),
        (1.3, 0.5, 0.8),
        (1.7, 0.7, 0.7),
        (1.7, 0.3, 0.5),
        (1.7, 0.3, 0.25)
    ]

    points_ready = [
        (1.7, 0.3, 0.25),
        (1.7, 0.3, 0.20),
        (1.7, 0.3, 0.15),
        (1.7, 0.3, 0.11),
    ]
    points_after = [
        (1.7, 0.3, 0.11),
        (1.3, 0.5, 0.3),
        (1.3, 0.7, 0.4),
        (1.3, 0.7, 0.5),
    ]

    smooth_before = smooth_drone_path(points_before, num_points=150, s=0.2)
    smooth_ready = smooth_drone_path(points_ready, num_points=30, s=0.0, dim=1)
    smooth_after = smooth_drone_path(points_after, num_points=100, s=0.2)

    mass_guess_done = False
    new_mass = 0
    pickup_log_start_idx = None
    stabilize_log_end_idx = None

    for p in smooth_before:
        fly_to_point(p, controller, scene, cam, 15, 0, ctx)

        if abs(p[2] - points_before[2][2]) < 0.01 and not mass_guess_done:  # if z is close to 0.3, do mass guess, for only once
            coke_pos = CokeCan.get_pos().cpu().numpy().reshape(-1)
            cam2_pos = (float(coke_pos[0] - 0.35), float(coke_pos[1]), float(coke_pos[2] + 0.20))
            cam2_lookat = (float(coke_pos[0]), float(coke_pos[1]), float(coke_pos[2] + 0.05))
            cam2.set_pose(pos=cam2_pos, lookat=cam2_lookat)

            rgb_out = safe_render(cam2)
            if rgb_out is None:
                continue
            rgb, _, _, _ = rgb_out
            # new_mass = guess_mass(rgb)
            # print("[mass guess]", new_mass)
            mass_guess_done = True

    for i, p in enumerate(smooth_ready):
        tstep = waypoint_step_budget(drone, p, base=22, gain=140.0, min_steps=22, max_steps=200)
        d_last = fly_to_point(p, controller, scene, cam, tstep, 0, ctx)

        # strict pre-pick convergence on the final approach point
        if i == len(smooth_ready) - 1:
            retry = 0
            while d_last > 0.005 and retry < 4:
                d_last = fly_to_point(p, controller, scene, cam, 120, 0, ctx)
                retry += 1
            print(f"[pre-pick] final distance={d_last:.4f} m after {retry} retries")
        fly_to_point(p, controller, scene, cam, 10, 0, ctx)
    
    # fly_to_point(points_ready[-1], controller, scene, cam, 1000, 0, ctx)  # directly fly to the final pre-pick point (no retries, for simplicity)

    # Use guess to initialize theta_hat (your CP step will later give interval too)
    # ctx.theta_est.reset(float(0.05))

        
    end_effector = drone.get_link("gripper_base")  # or any gripper link; they are welded together so it shouldn't matter much
    can_link = CokeCan.base_link

    # Create constraint arrays
    # quaternion
    drone_quat = end_effector.get_quat().cpu().numpy().reshape(-1)
    drone_pos = end_effector.get_pos().cpu().numpy().reshape(-1)

    # Position alignment
    # CokeCan.set_quat((float(drone_quat[0]), float(drone_quat[1]), float(drone_quat[2]), float(drone_quat[3])))
    CokeCan.set_pos((float(drone_pos[0]), float(drone_pos[1]), CokeCan.get_pos().cpu().numpy().reshape(-1)[2]))  # z는 원래 위치 유지

    link_can = np.array([can_link.idx], dtype=gs.np_int)
    link_drone = np.array([end_effector.idx], dtype=gs.np_int)
    rigid.add_weld_constraint(link_can, link_drone)

    pick_up_box(scene, drone, cam, cam2, controller, ctx, points_ready[-1])
    pickup_log_start_idx = len(ctx.log["t"])
    ctx.theta_est.reset(float(new_mass/100))
    # ctx.theta_est.reset(float(0.0035))
    
    # Clear controller states after grasp to avoid carrying pre-pick integral bias.
    controller.reset_pid_states()

    # Keep XY adaptation off during post-pick stabilization to avoid learning transient disturbance.
    controller.reset_adaptive_xy(0.0, 0.0)
    controller.set_adaptive_xy_enabled(False)

    # After grasp: hover and wait until speed settles, then start transport path.
    stabilize_after_pickup(
        scene=scene,
        drone=drone,
        controller=controller,
        cam=cam,
        ctx=ctx,
        hold_target=tuple(map(float, smooth_after[0])),
        hover_up=0.08,
        max_steps=600,
        speed_tol=0.06,
        stable_steps_required=120,
        min_hover_steps=220,
    )
    stabilize_log_end_idx = len(ctx.log["t"])

    # Reset once more at transport start and then enable XY adaptation.
    controller.reset_pid_states()
    # controller.reset_adaptive_xy(0.0, 0.0)
    controller.set_adaptive_xy_enabled(True)

    # NOTE: you currently do controller.change_theta(65.5)
    # If you keep that, it may "double count" with adaptive thrust from theta_hat.
    # Consider disabling it once adaptive is active.
    # controller.change_theta(65.5)

    for p in smooth_after:
        fly_to_point(p, controller, scene, cam, 15, 0, ctx)

    rigid.delete_weld_constraint(link_can, link_drone)
    drop_box(scene, drone, cam, controller)

    cam.stop_recording(save_to_filename="videos/fly_route_robust_adaptive.mp4", fps=100)

    # save logs
    np.savez("robust_adaptive_logs.npz", **{k: np.array(v) for k, v in ctx.log.items()})
    print("Saved robust_adaptive_logs.npz")

    # quick diagnostic, log only after drone pick up

    try:
        t = np.array(ctx.log["t"])
        x = np.array(ctx.log["x"])
        y = np.array(ctx.log["y"])
        z = np.array(ctx.log["z"])
        xt = np.array(ctx.log["target_x"])
        yt = np.array(ctx.log["target_y"])
        zt = np.array(ctx.log["target_z"])
        sqrtV = np.array(ctx.log["sqrtV"])
        Rbar = np.array(ctx.log["Rbar"])
        score = np.array(ctx.log["score"])
        theta_hat = np.array(ctx.log["theta_hat"])
        margin = np.array(ctx.log["margin"])
        V = np.array(ctx.log["V"])

        def _apply_readable_axis_interval(values: np.ndarray, pad_ratio: float = 0.15, min_span: float = 0.02):
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size == 0:
                return None
            vmin = float(np.min(finite_vals))
            vmax = float(np.max(finite_vals))
            span = max(vmax - vmin, min_span)
            pad = pad_ratio * span
            mid = 0.5 * (vmin + vmax)
            half = 0.5 * span + pad
            return (mid - half, mid + half)

        plt.figure(); plt.plot(t[1000:], x[1000:], label="x"); plt.plot(t[1000:], xt[1000:], "--", label="x_target"); plt.legend(); plt.grid(True); plt.title("x tracking"); plt.xlabel("Time [s]"); plt.ylabel("x [m]")
        plt.savefig("plot_x_tracking.png", dpi=200); plt.close()

        plt.figure(); plt.plot(t[1000:], y[1000:], label="y"); plt.plot(t[1000:], yt[1000:], "--", label="y_target"); plt.legend(); plt.grid(True); plt.title("y tracking"); plt.xlabel("Time [s]"); plt.ylabel("y [m]")
        plt.savefig("plot_y_tracking.png", dpi=200); plt.close()

        plt.figure(); plt.plot(t[1000:], z[1000:], label="z"); plt.plot(t[1000:], zt[1000:], "--", label="z_target"); plt.legend(); plt.grid(True); plt.title("z tracking"); plt.xlabel("Time [s]"); plt.ylabel("z [m]")
        plt.savefig("plot_z_tracking.png", dpi=200); plt.close()

        if (
            pickup_log_start_idx is not None
            and stabilize_log_end_idx is not None
            and 0 <= pickup_log_start_idx < stabilize_log_end_idx <= len(t)
        ):
            sl = slice(pickup_log_start_idx, stabilize_log_end_idx)
            t_sl = t[sl]
            z_sl = z[sl]
            zt_sl = zt[sl]
            V_sl = V[sl]
            Rbar_sl = Rbar[sl]
            margin_sl = margin[sl]

            z_ylim = _apply_readable_axis_interval(np.concatenate([z_sl, zt_sl]), pad_ratio=0.25, min_span=0.05)
            V_ylim = _apply_readable_axis_interval(V_sl, pad_ratio=0.25, min_span=0.05)

            plt.figure(); plt.plot(t_sl, z_sl, label="z"); plt.plot(t_sl, zt_sl, "--", label="z_target"); plt.legend(); plt.grid(True); plt.title("z tracking (pickup to stabilization)"); plt.xlabel("Time [s]"); plt.ylabel("z [m]")
            if z_ylim is not None:
                plt.ylim(z_ylim)
            plt.savefig("plot_z_pickup_to_stabilization.png", dpi=200); plt.close()

            plt.figure(); plt.plot(t_sl, V_sl, label="V"); plt.legend(); plt.grid(True); plt.title("V (pickup to stabilization)"); plt.xlabel("Time [s]"); plt.ylabel("V")
            if V_ylim is not None:
                plt.ylim(V_ylim)
            plt.savefig("plot_V_pickup_to_stabilization.png", dpi=200); plt.close()

            plt.figure(); plt.plot(t_sl, Rbar_sl, label="R(ACP bound)"); plt.legend(); plt.grid(True); plt.title("R(ACP bound) (pickup to stabilization)"); plt.xlabel("Time [s]"); plt.ylabel("R(ACP bound)")
            plt.savefig("plot_Rbar_pickup_to_stabilization.png", dpi=200); plt.close()

            plt.figure(); plt.plot(t_sl, margin_sl, label="margin"); plt.axhline(0.0, linestyle="--"); plt.legend(); plt.grid(True); plt.title("CLF margin (pickup to stabilization)"); plt.xlabel("Time [s]"); plt.ylabel("margin")
            plt.savefig("plot_margin_pickup_to_stabilization.png", dpi=200); plt.close()

        plt.figure(); plt.plot(t[1000:], sqrtV[1000:], label="sqrt(V)"); plt.legend(); plt.grid(True); plt.title("sqrt(V)"); plt.xlabel("Time [s]"); plt.ylabel("sqrt(V)")
        plt.savefig("plot_sqrtV.png", dpi=200); plt.close()

        plt.figure(); plt.plot(t, np.abs(score), label="|score|"); plt.plot(t, Rbar, label="R(ACP bound)"); plt.legend(); plt.grid(True); plt.title("ACP bound"); plt.xlabel("Time [s]");   plt.ylabel("Bound / score")
        plt.savefig("plot_Rbar.png", dpi=200); plt.close()

        # Conformal diagnostics: interval violations and empirical coverage over time.
        abs_score = np.abs(score)
        valid_mask = np.isfinite(abs_score) & np.isfinite(Rbar)
        violation = np.zeros_like(abs_score, dtype=bool)
        violation[valid_mask] = abs_score[valid_mask] > Rbar[valid_mask]

        plt.figure()
        plt.plot(t, abs_score, label="|score|", linewidth=1.2)
        plt.plot(t, Rbar, label="R(ACP bound)", linewidth=1.2)
        if np.any(violation):
            plt.scatter(t[violation], abs_score[violation], s=8, c="red", alpha=0.7, label="violations")
        plt.legend()
        plt.grid(True)
        plt.title("Conformal score vs bound")
        plt.xlabel("Time [s]")
        plt.ylabel("Score / bound")
        plt.savefig("plot_cp_score_bound.png", dpi=200)
        plt.close()

        covered = np.zeros_like(abs_score, dtype=float)
        covered[valid_mask] = (abs_score[valid_mask] <= Rbar[valid_mask]).astype(float)
        valid_counts = np.cumsum(valid_mask.astype(float))
        running_coverage = np.divide(
            np.cumsum(covered),
            np.maximum(valid_counts, 1.0),
        )
        target_coverage = 1.0 - float(aci.alpha_target)

        plt.figure()
        plt.plot(t, running_coverage, label="running empirical coverage", linewidth=1.4)
        plt.axhline(target_coverage, linestyle="--", color="black", label=f"target 1-alpha = {target_coverage:.2f}")
        plt.ylim(0.0, 1.02)
        plt.legend()
        plt.grid(True)
        plt.title("Conformal running coverage")
        plt.xlabel("Time [s]")
        plt.ylabel("Coverage")
        plt.savefig("plot_cp_running_coverage.png", dpi=200)
        plt.close()

        plt.figure(); plt.plot(t, theta_hat, label="theta_hat"); plt.legend(); plt.grid(True); plt.title("theta_hat"); plt.xlabel("Time [s]"); plt.ylabel("theta_hat")
        plt.savefig("plot_theta_hat.png", dpi=200); plt.close()

        plt.figure(); plt.plot(t, margin, label="margin"); plt.axhline(0.0, linestyle="--"); plt.legend(); plt.grid(True); plt.title("CLF margin (<=0 desired)"); plt.xlabel("Time [s]"); plt.ylabel("margin")
        plt.savefig("plot_margin.png", dpi=200); plt.close()

        print("Saved plots: plot_x_tracking.png, plot_y_tracking.png, plot_z_tracking.png, plot_sqrtV.png, plot_Rbar.png, plot_cp_score_bound.png, plot_cp_running_coverage.png, plot_theta_hat.png, plot_margin.png, plot_z_pickup_to_stabilization.png, plot_V_pickup_to_stabilization.png, plot_Rbar_pickup_to_stabilization.png, plot_margin_pickup_to_stabilization.png")
    except Exception as e:
        print("Plotting failed:", e)

if __name__ == "__main__":
    main()
