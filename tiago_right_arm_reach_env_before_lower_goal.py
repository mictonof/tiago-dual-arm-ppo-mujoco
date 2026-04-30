import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description


class TiagoRightArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        max_steps=300,
        frame_skip=10,
        action_scale=0.002,
        success_threshold=0.05,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.success_threshold = success_threshold

        self.model = load_robot_description("tiago++_mj_description")
        self.data = mujoco.MjData(self.model)

        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        self.reference_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "reference"
        )
        self.reference_qpos_adr = self.model.jnt_qposadr[self.reference_joint_id]
        self.reference_dof_adr = self.model.jnt_dofadr[self.reference_joint_id]
        self.reference_qpos0 = self.init_qpos[
            self.reference_qpos_adr : self.reference_qpos_adr + 7
        ].copy()

        self.right_joint_names = [
            "arm_right_1_joint",
            "arm_right_2_joint",
            "arm_right_3_joint",
            "arm_right_4_joint",
            "arm_right_5_joint",
            "arm_right_6_joint",
            "arm_right_7_joint",
        ]
        self.right_actuator_names = [
            "arm_right_1_joint_position",
            "arm_right_2_joint_position",
            "arm_right_3_joint_position",
            "arm_right_4_joint_position",
            "arm_right_5_joint_position",
            "arm_right_6_joint_position",
            "arm_right_7_joint_position",
        ]
        self.left_actuator_names = [
            "arm_left_1_joint_position",
            "arm_left_2_joint_position",
            "arm_left_3_joint_position",
            "arm_left_4_joint_position",
            "arm_left_5_joint_position",
            "arm_left_6_joint_position",
            "arm_left_7_joint_position",
        ]

        self.right_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.right_joint_names
        ]
        self.right_qpos_idx = [self.model.jnt_qposadr[jid] for jid in self.right_joint_ids]
        self.right_qvel_idx = [self.model.jnt_dofadr[jid] for jid in self.right_joint_ids]

        self.right_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.right_actuator_names
        ]
        self.left_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.left_actuator_names
        ]

        self.right_finger_body_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_right_finger_link"
            ),
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_left_finger_link"
            ),
        ]

        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_lift_link"
        )
        self.right_shoulder_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "arm_right_1_link"
        )

        self.ctrl_home = np.zeros(self.model.nu, dtype=np.float64)
        for aid in range(self.model.nu):
            aname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            if aname is None:
                continue
            if aname.endswith("_position"):
                jname = aname.replace("_position", "")
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                qidx = self.model.jnt_qposadr[jid]
                self.ctrl_home[aid] = self.init_qpos[qidx]
            else:
                self.ctrl_home[aid] = 0.0

        # Brazos recogidos
        self.right_home_pose = np.array([0.10, 1.00, 1.30, 1.55, -1.15, 1.00, 0.00], dtype=np.float64)
        self.left_home_pose  = np.array([0.10, 1.00, 1.30, 1.55, -1.15, 1.00, 0.00], dtype=np.float64)

        self.ctrl = self.ctrl_home.copy()
        self._apply_compact_home_targets()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

        self.viewer = None
        self.goal = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.prev_dist = None

    def _apply_compact_home_targets(self):
        for i, aid in enumerate(self.right_actuator_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.ctrl_home[aid] = np.clip(self.right_home_pose[i], lo, hi)

        for i, aid in enumerate(self.left_actuator_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.ctrl_home[aid] = np.clip(self.left_home_pose[i], lo, hi)

    def _hold_reference(self):
        self.data.qpos[
            self.reference_qpos_adr : self.reference_qpos_adr + 7
        ] = self.reference_qpos0
        self.data.qvel[
            self.reference_dof_adr : self.reference_dof_adr + 6
        ] = 0.0

    def _ee_pos(self):
        p1 = self.data.xpos[self.right_finger_body_ids[0]].copy()
        p2 = self.data.xpos[self.right_finger_body_ids[1]].copy()
        return 0.5 * (p1 + p2)

    def _get_obs(self):
        q = np.array([self.data.qpos[i] for i in self.right_qpos_idx], dtype=np.float32)
        dq = np.array([self.data.qvel[i] for i in self.right_qvel_idx], dtype=np.float32)
        ee = self._ee_pos().astype(np.float32)
        goal = self.goal.astype(np.float32)
        err = goal - ee
        obs = np.concatenate([q, dq, ee, goal, err]).astype(np.float32)
        dist = float(np.linalg.norm(err))
        return obs, dist

    def _sample_goal(self):
        ee0 = self._ee_pos().copy()
        torso = self.data.xpos[self.torso_body_id].copy()
        shoulder = self.data.xpos[self.right_shoulder_body_id].copy()

        # Misma distribución que te gustó, pero bastante más abajo
        zones = ["front_right_mid", "front_right_far", "from_ee_right"]
        probs = [0.45, 0.35, 0.20]

        for _ in range(500):
            zone = self.np_random.choice(zones, p=probs)

            if zone == "front_right_mid":
                g = torso + np.array([
                    self.np_random.uniform(0.26, 0.33),
                    self.np_random.uniform(-0.18, -0.08),
                    self.np_random.uniform(-0.26, -0.14),
                ])
            elif zone == "front_right_far":
                g = torso + np.array([
                    self.np_random.uniform(0.30, 0.36),
                    self.np_random.uniform(-0.20, -0.10),
                    self.np_random.uniform(-0.28, -0.16),
                ])
            else:
                g = ee0 + np.array([
                    self.np_random.uniform(0.14, 0.20),
                    self.np_random.uniform(-0.12, -0.05),
                    self.np_random.uniform(-0.20, -0.10),
                ])

            dist_ee = np.linalg.norm(g - ee0)
            dist_shoulder = np.linalg.norm(g - shoulder)
            dist_torso = np.linalg.norm(g - torso)

            if not (0.14 <= dist_ee <= 0.24):
                continue
            if not (0.22 <= dist_shoulder <= 0.44):
                continue
            if not (0.24 <= dist_torso <= 0.38):
                continue
            if not (0.35 <= g[2] <= 0.80):
                continue

            if g[0] <= torso[0] + 0.10:
                continue
            if g[1] >= torso[1] - 0.03:
                continue

            return g.astype(np.float64)

        return (torso + np.array([0.30, -0.12, -0.18], dtype=np.float64)).astype(np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel

        self.ctrl = self.ctrl_home.copy()
        self.data.ctrl[:] = self.ctrl

        self._hold_reference()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(80):
            self.data.ctrl[:] = self.ctrl
            mujoco.mj_step(self.model, self.data)
            self._hold_reference()
            mujoco.mj_forward(self.model, self.data)

        self.goal = self._sample_goal()

        obs, dist = self._get_obs()
        self.prev_dist = dist

        if self.render_mode == "human":
            self.render()

        return obs, {"goal": self.goal.copy()}

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -0.5, 0.5).astype(np.float64)

        current_targets = self.ctrl[self.right_actuator_ids].copy()

        delta = action * self.action_scale
        max_delta = 0.002
        delta = np.clip(delta, -max_delta, max_delta)
        new_targets = current_targets + delta

        for k, aid in enumerate(self.right_actuator_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            new_targets[k] = np.clip(new_targets[k], lo, hi)

        self.ctrl[self.right_actuator_ids] = new_targets

        prev_dist = self.prev_dist

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.ctrl
            mujoco.mj_step(self.model, self.data)
            self._hold_reference()
            mujoco.mj_forward(self.model, self.data)

        obs, dist = self._get_obs()
        self.prev_dist = dist

        progress = prev_dist - dist
        reward = -1.5 * dist + 6.0 * progress - 0.0005 * float(np.sum(action ** 2))

        success = dist < self.success_threshold
        timeout = self.step_count >= self.max_steps

        if success:
            reward += 10.0

        terminated = success
        truncated = timeout

        info = {
            "distance_to_goal": dist,
            "is_success": success,
            "goal": self.goal.copy(),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )

        ee = self._ee_pos()
        look = 0.5 * (ee + self.goal)

        with self.viewer.lock():
            self.viewer.cam.lookat[:] = [look[0], look[1], look[2]]
            self.viewer.cam.distance = 1.8
            self.viewer.cam.azimuth = 160
            self.viewer.cam.elevation = -18

            self.viewer.user_scn.ngeom = 3

            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.03, 4.0, 3.0]),
                pos=np.array([-1.8, 0.0, 1.5]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 1.0, 1.0, 1.0]),
            )

            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([3.0, 3.0, 0.02]),
                pos=np.array([0.0, 0.0, -0.02]),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.96, 0.96, 0.96, 1.0]),
            )

            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.025, 0.0, 0.0]),
                pos=self.goal,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.1, 0.1, 1.0]),
            )

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
