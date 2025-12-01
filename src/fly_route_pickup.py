import genesis as gs
import math
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import re
from scipy.spatial.transform import Rotation as R
from quadcopter_controller_AC import DronePIDController
from genesis.engine.entities.drone_entity import DroneEntity
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.entities.rigid_entity import RigidJoint
from genesis.vis.camera import Camera
from genesis.options.morphs import Primitive
from PIL import Image

from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
model_path = "lmms-lab/LLaVA-One-Vision-1.5-8B-Instruct"


# handbag urdf
# drone control balance
# If mass > one drone capacity, more drone join to do task

# base_rpm = (14468.429183500699 + 533 - 10)*np.sqrt(59.82/29)
base_rpm = 14468.429183500699 * np.sqrt(2.595)
min_rpm = 0.9 * base_rpm
max_rpm = 1.5 * base_rpm

pick_up_1 = 0

view_0 = (0, 0, 0)
view_1 = (0, 0, 1)

def guess_mass(rgb):
    
    # (1) NumPy array → PIL.Image.Image generate
    pil_img = Image.fromarray(rgb)

    # (2) Store as temp file
    temp_path = "temp_image.png"
    pil_img.save(temp_path)    

    # default: Load the model on the available device(s)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": temp_path,
                },
                {"type": "text", "text": "Give me the weight of the object in the sight only with scalar value"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

      # Only number
    numbers = re.findall(r'\d+\.?\d*', str(output_text))
    numeric_values = [float(x) if ('.' in x) else int(x) for x in numbers]

    print(numbers)
    print(numeric_values)

    new_mass = numeric_values

    new_mass = 0
    return new_mass

def hover(drone: DroneEntity, base_rpm = base_rpm):
    drone.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])


def clamp(rpm):
    return max(min_rpm, min(int(rpm), max_rpm))

def pick_up_box(scene: gs.Scene, drone: DroneEntity, cam: Camera, cam2: Camera, controller: DronePIDController):
    print('pick up')
    left = drone.get_joint("joint_left_finger")
    right = drone.get_joint("joint_right_finger")
    left_idx  = left.dof_idx
    right_idx = right.dof_idx
    
    up = drone.get_joint("joint_up_finger")
    down = drone.get_joint("joint_down_finger")
    up_idx  = up.dof_idx
    down_idx = down.dof_idx
    gripper_dofs = [left_idx, right_idx, up_idx, down_idx]
    
    '''
    left_d = drone.get_joint("joint_left_finger_d")
    right_d = drone.get_joint("joint_right_finger_d")
    left_idx_d  = left_d.dof_idx
    right_idx_d = right_d.dof_idx
    
    up_d = drone.get_joint("joint_up_finger_d")
    down_d = drone.get_joint("joint_down_finger_d")
    up_idx_d  = up_d.dof_idx
    down_idx_d = down_d.dof_idx
    gripper_dofs_d = [left_idx_d, right_idx_d, up_idx_d, down_idx_d]

    
    drone.control_dofs_position(
        position=[0.01, 0.01],
        dofs_idx_local=gripper_dofs
    )
    
    for i in range(10):
        hover(drone)
        scene.step()
        cam.render()
        pos = drone.get_pos().cpu().numpy()
        cam.set_pose(lookat=view_0)
        # cam.move_to_attach()

    drone.control_dofs_force(
        force = np.array([-1, 1]),
        dofs_idx_local=gripper_dofs
    )
    '''

    for i in range(5):
        hover(drone)
        scene.step()
        cam.render()
        pos = drone.get_pos().cpu().numpy()
        cam.set_pose(lookat=view_0)
        # cam.move_to_attach()  

    drone.control_dofs_force(
        force = np.array([0.5,0.5,0.5,0.5]),
        dofs_idx_local=gripper_dofs
    )
    

    for i in range(40):
        hover(drone)
        scene.step()
        cam.render()
        pos = drone.get_pos().cpu().numpy()
        cam.set_pose(lookat=view_0)
        # cam.move_to_attach()

    # controller.change_theta(151)
    # controller.change_base_rpm(base_rpm*np.sqrt(62/59.83))
    
    for i in range(20):
        hover(drone, base_rpm*np.sqrt(800/420))
        scene.step()
        cam.render()
        pos = drone.get_pos().cpu().numpy()
        cam.set_pose(lookat=view_0)
        # cam.move_to_attach()

    drone.control_dofs_force(
        force = np.array([0.7,0.7,0.7,0.7]),
        dofs_idx_local=gripper_dofs
    )
    
    # controller.change_base_rpm(base_rpm*np.sqrt(310/290))

def drop_box(scene: gs.Scene, drone: DroneEntity, cam: Camera, controller: DronePIDController):
    print('drop')

    left = drone.get_joint("joint_left_finger")
    right = drone.get_joint("joint_right_finger")
    left_idx  = left.dof_idx
    right_idx = right.dof_idx
    
    up = drone.get_joint("joint_up_finger")
    down = drone.get_joint("joint_down_finger")
    up_idx  = up.dof_idx
    down_idx = down.dof_idx
    gripper_dofs = [left_idx, right_idx, up_idx, down_idx]
    
    '''
    left_d = drone.get_joint("joint_left_finger_d")
    right_d = drone.get_joint("joint_right_finger_d")
    left_idx_d  = left_d.dof_idx
    right_idx_d = right_d.dof_idx
    
    up_d = drone.get_joint("joint_up_finger_d")
    down_d = drone.get_joint("joint_down_finger_d")
    up_idx_d  = up_d.dof_idx
    down_idx_d = down_d.dof_idx
    gripper_dofs_d = [left_idx_d, right_idx_d, up_idx_d, down_idx_d]
    '''

    drone.control_dofs_position(
        np.array([0,0,0,0]),
        dofs_idx_local=gripper_dofs
    )

    for i in range(150):
        hover(drone)
        scene.step()
        cam.render()
        pos = drone.get_pos().cpu().numpy()
        cam.set_pose(lookat=view_0)
        # cam.move_to_attach()

    # guess_mass()
    # controller.change_theta(0.035)

def fly_to_point(target, controller: DronePIDController, scene: gs.Scene, cam: Camera, Tstep: float, Fig: float):
    drone = controller.drone
    step = 0
    x = target[0] - drone.get_pos()[0]
    y = target[1] - drone.get_pos()[1]
    z = target[2] - drone.get_pos()[2]

    distance = math.sqrt(x**2 + y**2 + z**2)

    # rgb, _, _, _ = cam2.render()

    ############ theta initial guess transient comparision ################

    # desired position
    desired_x = 1.7
    desired_y = 0.3
    desired_z = 0.8

    # History list
    x_list, y_list, z_list = [], [], []
    time_list = []

    while (distance > 0.001) and step < Tstep:

        [M1, M2, M3, M4] = controller.update(target)
        M1 = clamp(M1)
        M2 = clamp(M2)
        M3 = clamp(M3)
        M4 = clamp(M4)
        drone.set_propellels_rpm([M1, M2, M3, M4])
        scene.step()
        cam.render()
        # print("point =", drone.get_pos())
        pos = drone.get_pos().cpu().numpy()
        cam.set_pose(lookat=view_0)
        # cam.move_to_attach()
        distance = math.dist(pos, target)
        print(pos)
        step += 1
        
        if Fig == 1:
            x_list.append(pos[0])
            y_list.append(pos[1])
            z_list.append(pos[2])
            time_list.append(step/100)

    if Fig == 1:
        plt.figure(figsize=(10, 6))

        # x
        plt.plot(time_list, x_list, label='x position (actual)', color = 'red', linestyle='-')
        plt.hlines(desired_x, time_list[0], time_list[-1], colors='r', linestyles='--', label='x desired')

        # y
        plt.plot(time_list, y_list, label='y position (actual)', color = 'green', linestyle='-')
        plt.hlines(desired_y, time_list[0], time_list[-1], colors='g', linestyles='--', label='y desired')

        # z
        plt.plot(time_list, z_list, label='z position (actual)', color = 'blue', linestyle='-')
        plt.hlines(desired_z, time_list[0], time_list[-1], colors='b', linestyles='--', label='z desired')

        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.title('Drone Position vs Fixed Target Position')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('drone_fixed_target_plot.png', dpi=300)
        plt.close()


def main():
    gs.init(backend=gs.gpu)

    ##### scene #####
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))

    ##### entities #####
    plane_0 = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    plane_1 = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane_light.urdf", euler = (0,90,45), fixed=True))

    drone = scene.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x_gripper3.urdf", pos=(0.5, 0.5, 0.5), scale = 2.3))

    bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=25),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.05,
            euler=(0, 0, 90),
            pos=(-0.4, 0.9, 0.6),
            fixed = True
        ),
        # visualize_contact=True,
    )

    Duck = scene.add_entity(gs.morphs.URDF(file="urdf/data/duck_vhacd.urdf", scale = 1, euler = (90,0,0), pos=(0.9, 0.2, 0.02), fixed=True))
    Teddy = scene.add_entity(gs.morphs.URDF(file="urdf/data/teddy_vhacd.urdf", scale = 1, euler = (90,0,0), pos=(0.6, 0, 0.02), fixed=True))
    Racecar = scene.add_entity(gs.morphs.URDF(file="urdf/data/racecar/racecar.urdf", scale = 0.3, pos=(0.7, -0.1, 0.02), fixed=True))
    Lego = scene.add_entity(gs.morphs.URDF(file="urdf/data/lego/lego.urdf", scale = 1, pos=(1, 0.3, 0.02), fixed=False))

    Table = scene.add_entity(gs.morphs.URDF(file="urdf/data/table/table.urdf", scale = 0.5, euler = (0,0,90), pos=(-0.3, 0.8, 0.25), fixed=True))
    Jenga = scene.add_entity(gs.morphs.URDF(file="urdf/data/jenga/jenga.urdf", scale = 0.5, pos=(-0.3, 0.8, 0.6), fixed=True))
    Pan = scene.add_entity(gs.morphs.URDF(file="urdf/data/dinnerware/pan_tefal.urdf", scale = 0.5, pos=(-0.2, 0.7, 0.6), fixed=True))
    Plate = scene.add_entity(gs.morphs.URDF(file="urdf/data/dinnerware/plate.urdf", scale = 0.5, pos=(-0.4, 0.7, 0.6), fixed=True))
    CokeCan = scene.add_entity(gs.morphs.URDF(file="urdf/coke_can/coke_can.urdf", scale = 0.5, pos=(1.7, 0.3, 0.05), fixed=False)) 
    
    pid_params = [
        [1.0, 0.0, 0],
        [1.0, 0.0, 0],
        [1.0, 0.0, 0],
        [8.0, 0.0, 3.0],
        [8.0, 0.0, 3.0],
        [10.0, 2.0, 2.0],
        [5.0, 0.0, 1.0],
        [5.0, 0.0, 1.0],
        [2.0, 0.0, 0.2],
    ]
    
    
    controller = DronePIDController(drone=drone, dt=0.01, base_rpm=base_rpm, pid_params=pid_params, gamma = 0.1, gamma_a = 0.01, nominal_mass = 150)

    cam = scene.add_camera(pos=(3,3,2), lookat=view_0, GUI=False, res=(1280, 960), fov=30)
    cam2 = scene.add_camera(pos=(2,2,1), lookat=view_0, GUI=False, res=(1280, 960), fov=30)


    ##### build #####

    scene.build()

    base_link = drone.get_link("base_link")
    base_link.set_mass(0.05)

    left = drone.get_link("gripper_left")
    right = drone.get_link("gripper_right")
    up = drone.get_link("gripper_up")
    down = drone.get_link("gripper_down")
    left.set_mass(0.005)
    right.set_mass(0.005)
    up.set_mass(0.005)
    down.set_mass(0.005)

    camera_forward = np.array([1, 0, 0])  

    target_dir = np.array([0.7, 0.0, 1])

    rot, _ = R.align_vectors(
        np.vstack([target_dir]),     
        np.vstack([camera_forward])  
    )

    offset_R = rot.as_matrix()   
    offset_T = np.eye(4)
    offset_T[:3, :3] = offset_R

    cam2.attach(right, offset_T = offset_T)

    cam.start_recording()

    points = [(0.9, 0.3, 0.6), (1.3, 0.3, 0.3), (1.7, 0.3, 0.6), (1.7, 0.3, 0.3), (1.7, 0.3, 0.11), (1.7, 0.3, 0.8), (0.1, 0.9, 1.0)]   

    fly_to_point(points[0], controller, scene, cam, 1000, 0)
    fly_to_point(points[1], controller, scene, cam, 1000, 0)

    cam2.move_to_attach()
    rgb, _, _, _ = cam2.render()
    new_mass = guess_mass(rgb)  

    fly_to_point(points[2], controller, scene, cam, 1000, 0)
    fly_to_point(points[3], controller, scene, cam, 1000, 0)
    fly_to_point(points[4], controller, scene, cam, 1000, 0)

    pick_up_box(scene, drone, cam, cam2, controller)
    controller.change_theta(97.5)

    fly_to_point(points[5], controller, scene, cam, 3000, 1)
    drop_box(scene, drone, cam, controller)

    cam.stop_recording(save_to_filename="videos/fly_route_151_cup.mp4", fps=100)

if __name__ == "__main__":
    main()
