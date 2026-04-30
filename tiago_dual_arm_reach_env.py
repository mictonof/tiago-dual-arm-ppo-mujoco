import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description


class TiagoDualArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        max_steps=350,
        frame_skip=10,
        action_scale=0.0075,
        success_threshold=0.05,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.success_threshold = success_threshold

        self.goal_radius = 0.018
        self.ee_visual_radius = 0.032
        self.goal_clearance = 0.015

        # Prisma movido más hacia delante
        self.prism_offset = np.array([0.43, 0.00, -0.08], dtype=np.float64)
        self.prism_half_extents = np.array([0.075, 0.145, 0.140], dtype=np.float64)

        self.min_goal_dist = 0.08
        self.max_goal_dist = 0.34
        self.max_action_delta = 0.0075

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

        self.sides = ("right", "left")
        self.side_sign = {"right": -1.0, "left": 1.0}

        self.joint_names = {
            side: [f"arm_{side}_{i}_joint" for i in range(1, 8)]
            for side in self.sides
        }
        self.actuator_names = {
            side: [f"arm_{side}_{i}_joint_position" for i in range(1, 8)]
            for side in self.sides
        }
        self.gripper_actuator_names = {
            side: [
                f"gripper_{side}_left_finger_joint_position",
                f"gripper_{side}_right_finger_joint_position",
            ]
            for side in self.sides
        }

        self.joint_ids = {
            side: [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.joint_names[side]
            ]
            for side in self.sides
        }
        self.qpos_idx = {
            side: [self.model.jnt_qposadr[jid] for jid in self.joint_ids[side]]
            for side in self.sides
        }
        self.qvel_idx = {
            side: [self.model.jnt_dofadr[jid] for jid in self.joint_ids[side]]
            for side in self.sides
        }
        self.actuator_ids = {
            side: [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.actuator_names[side]
            ]
            for side in self.sides
        }
        self.gripper_actuator_ids = {
            side: [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.gripper_actuator_names[side]
            ]
            for side in self.sides
        }

        self.finger_body_ids = {
            side: [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"gripper_{side}_right_finger_link"
                ),
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"gripper_{side}_left_finger_link"
                ),
            ]
            for side in self.sides
        }

        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_lift_link"
        )
        self.wrist_body_id = {
            side: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"arm_{side}_7_link"
            )
            for side in self.sides
        }

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

        self.right_home_pose = np.array(
            [0.10, 1.00, 1.30, 1.55, -1.15, 1.00, 0.00], dtype=np.float64
        )
        self.left_home_pose = np.array(
            [0.10, 1.00, 1.30, 1.55, -1.15, 1.00, 0.00], dtype=np.float64
        )
        self.gripper_open = 0.03

        self.ctrl = self.ctrl_home.copy()
        self._apply_compact_home_targets()

        self.finger_subtrees = {
            side: [
                self._descendant_body_ids(self.finger_body_ids[side][0]),
                self._descendant_body_ids(self.finger_body_ids[side][1]),
            ]
            for side in self.sides
        }

        self.active_arm = "right"
        self.active_sign = -1.0
        self._bind_active_arm(self.active_arm)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

        self.viewer = None
        self.goal = np.zeros(3, dtype=np.float64)
        self.goal_visible = True
        self.step_count = 0
        self.prev_dist = None
        self._I3 = np.eye(3).flatten()

    def _bind_active_arm(self, arm):
        self.active_arm = arm
        self.active_sign = self.side_sign[arm]
        self.active_qpos_idx = self.qpos_idx[arm]
        self.active_qvel_idx = self.qvel_idx[arm]
        self.active_actuator_ids = self.actuator_ids[arm]

    def _descendant_body_ids(self, root_id):
        parent = self.model.body_parentid
        out = {int(root_id)}
        changed = True
        while changed:
            changed = False
            for bid in range(self.model.nbody):
                if int(parent[bid]) in out and bid not in out:
                    out.add(int(bid))
                    changed = True
        return sorted(out)

    def _apply_compact_home_targets(self):
        for i, aid in enumerate(self.actuator_ids["right"]):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.ctrl_home[aid] = np.clip(self.right_home_pose[i], lo, hi)

        for i, aid in enumerate(self.actuator_ids["left"]):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.ctrl_home[aid] = np.clip(self.left_home_pose[i], lo, hi)

        for aid in self.gripper_actuator_ids["right"] + self.gripper_actuator_ids["left"]:
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.ctrl_home[aid] = np.clip(self.gripper_open, lo, hi)

    def _hold_reference(self):
        self.data.qpos[self.reference_qpos_adr : self.reference_qpos_adr + 7] = self.reference_qpos0
        self.data.qvel[self.reference_dof_adr : self.reference_dof_adr + 6] = 0.0

    def _gripper_base_pos(self, side):
        return self.data.xpos[self.wrist_body_id[side]].copy()

    def _gripper_mid_pos(self, side):
        p1 = self.data.xpos[self.finger_body_ids[side][0]].copy()
        p2 = self.data.xpos[self.finger_body_ids[side][1]].copy()
        return 0.5 * (p1 + p2)

    def _gripper_axis(self, side):
        base = self._gripper_base_pos(side)
        mid = self._gripper_mid_pos(side)
        axis = mid - base
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis / norm

    def _farthest_geom_point_on_subtree(self, subtree_body_ids, side):
        axis = self._gripper_axis(side)
        base = self._gripper_base_pos(side)
        best_pos = None
        best_proj = -1e9

        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            if bid not in subtree_body_ids:
                continue
            pos = self.data.geom_xpos[gid].copy()
            proj = float(np.dot(pos - base, axis))
            if proj > best_proj:
                best_proj = proj
                best_pos = pos

        if best_pos is None:
            return self._gripper_mid_pos(side).copy()
        return best_pos

    def _gripper_action_center_pos(self, side):
        tip_a = self._farthest_geom_point_on_subtree(self.finger_subtrees[side][0], side)
        tip_b = self._farthest_geom_point_on_subtree(self.finger_subtrees[side][1], side)
        return 0.5 * (tip_a + tip_b)

    def _ee_pos(self, side=None):
        if side is None:
            side = self.active_arm
        return self._gripper_action_center_pos(side)

    def get_sampling_prism_center(self):
        torso = self.data.xpos[self.torso_body_id].copy()
        return torso + self.prism_offset

    def get_sampling_prism_half_extents(self):
        return self.prism_half_extents.copy()

    def _choose_closest_arm(self, goal):
        d_right = np.linalg.norm(goal - self._ee_pos("right"))
        d_left = np.linalg.norm(goal - self._ee_pos("left"))
        return ("right", d_right) if d_right <= d_left else ("left", d_left)

    def _sample_goal_and_arm(self):
        center = self.get_sampling_prism_center()
        half = self.get_sampling_prism_half_extents()

        best_goal = None
        best_arm = "right"
        best_dist = np.inf

        min_z = center[2] - half[2] + 0.05
        max_z = center[2] + half[2] - 0.06

        # recorta la parte trasera del prisma para que no quede pegada al robot
        min_x = center[0] - half[0] + 0.05

        for _ in range(300):
            goal = center + np.array([
                self.np_random.uniform(-half[0], half[0]),
                self.np_random.uniform(-half[1], half[1]),
                self.np_random.uniform(-half[2], half[2]),
            ], dtype=np.float64)

            if goal[2] < min_z or goal[2] > max_z:
                continue
            if goal[0] < min_x:
                continue

            arm, dist = self._choose_closest_arm(goal)

            if dist < best_dist:
                best_goal = goal.copy()
                best_arm = arm
                best_dist = dist

            if self.min_goal_dist <= dist <= self.max_goal_dist:
                return goal.astype(np.float64), arm

        return best_goal.astype(np.float64), best_arm

    def _get_obs(self):
        q = np.array([self.data.qpos[i] for i in self.active_qpos_idx], dtype=np.float32)
        dq = np.array([self.data.qvel[i] for i in self.active_qvel_idx], dtype=np.float32)
        ee = self._ee_pos().astype(np.float32)
        goal = self.goal.astype(np.float32)
        err = goal - ee
        arm_sign = np.array([self.active_sign], dtype=np.float32)
        obs = np.concatenate([q, dq, ee, goal, err, arm_sign]).astype(np.float32)
        dist = float(np.linalg.norm(err))
        return obs, dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.goal_visible = True

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self.ctrl = self.ctrl_home.copy()
        self.data.ctrl[:] = self.ctrl

        self._hold_reference()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(120):
            self.data.ctrl[:] = self.ctrl
            mujoco.mj_step(self.model, self.data)
            self._hold_reference()
            mujoco.mj_forward(self.model, self.data)

        self.goal, arm = self._sample_goal_and_arm()
        self._bind_active_arm(arm)

        obs, dist = self._get_obs()
        self.prev_dist = dist

        if self.render_mode == "human":
            self.render()

        return obs, {"goal": self.goal.copy(), "arm": self.active_arm}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        current_targets = self.ctrl[self.active_actuator_ids].copy()
        delta = np.clip(action * self.action_scale, -self.max_action_delta, self.max_action_delta)
        new_targets = current_targets + delta

        for k, aid in enumerate(self.active_actuator_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            new_targets[k] = np.clip(new_targets[k], lo, hi)

        self.ctrl[self.active_actuator_ids] = new_targets
        prev_dist = self.prev_dist

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.ctrl
            mujoco.mj_step(self.model, self.data)
            self._hold_reference()
            mujoco.mj_forward(self.model, self.data)

        obs, dist = self._get_obs()
        self.prev_dist = dist
        progress = prev_dist - dist

        reward = (
            -1.8 * dist
            + 25.0 * progress
            - 0.0001 * float(np.sum(action ** 2))
        )

        if dist < 0.09:
            reward += 1.5
        if dist < 0.065:
            reward += 2.0

        success = dist <= self.success_threshold
        timeout = self.step_count >= self.max_steps

        if success:
            reward += 70.0
            self.goal_visible = False
        elif timeout:
            reward -= 2.0

        terminated = success
        truncated = timeout

        info = {
            "distance_to_goal": dist,
            "is_success": success,
            "goal": self.goal.copy(),
            "arm": self.active_arm,
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

            self.viewer.user_scn.ngeom = 3 if self.goal_visible else 2

            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.03, 4.0, 3.0]),
                pos=np.array([-1.8, 0.0, 1.5]),
                mat=self._I3,
                rgba=np.array([1.0, 1.0, 1.0, 1.0]),
            )
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([3.0, 3.0, 0.02]),
                pos=np.array([0.0, 0.0, -0.02]),
                mat=self._I3,
                rgba=np.array([0.96, 0.96, 0.96, 1.0]),
            )

            if self.goal_visible:
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[2],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([self.goal_radius, 0.0, 0.0]),
                    pos=self.goal,
                    mat=self._I3,
                    rgba=np.array([1.0, 0.1, 0.1, 1.0]),
                )

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
