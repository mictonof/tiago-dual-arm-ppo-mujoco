import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description


class TiagoArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        arm="right",
        render_mode=None,
        max_steps=500,
        frame_skip=10,
        action_scale=0.004,
        success_threshold=0.07,
    ):
        super().__init__()

        if arm not in ("right", "left"):
            raise ValueError("arm debe ser 'right' o 'left'")

        self.arm = arm
        self.other_arm = "left" if arm == "right" else "right"
        self.side_sign = -1.0 if arm == "right" else 1.0

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.success_threshold = success_threshold

        self.goal_radius = 0.025
        self.goal_clearance = 0.03

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

        self.active_joint_names = [
            f"arm_{self.arm}_1_joint",
            f"arm_{self.arm}_2_joint",
            f"arm_{self.arm}_3_joint",
            f"arm_{self.arm}_4_joint",
            f"arm_{self.arm}_5_joint",
            f"arm_{self.arm}_6_joint",
            f"arm_{self.arm}_7_joint",
        ]
        self.active_actuator_names = [
            f"arm_{self.arm}_1_joint_position",
            f"arm_{self.arm}_2_joint_position",
            f"arm_{self.arm}_3_joint_position",
            f"arm_{self.arm}_4_joint_position",
            f"arm_{self.arm}_5_joint_position",
            f"arm_{self.arm}_6_joint_position",
            f"arm_{self.arm}_7_joint_position",
        ]

        self.other_actuator_names = [
            f"arm_{self.other_arm}_1_joint_position",
            f"arm_{self.other_arm}_2_joint_position",
            f"arm_{self.other_arm}_3_joint_position",
            f"arm_{self.other_arm}_4_joint_position",
            f"arm_{self.other_arm}_5_joint_position",
            f"arm_{self.other_arm}_6_joint_position",
            f"arm_{self.other_arm}_7_joint_position",
        ]

        self.active_gripper_actuator_names = [
            f"gripper_{self.arm}_left_finger_joint_position",
            f"gripper_{self.arm}_right_finger_joint_position",
        ]
        self.other_gripper_actuator_names = [
            f"gripper_{self.other_arm}_left_finger_joint_position",
            f"gripper_{self.other_arm}_right_finger_joint_position",
        ]

        self.active_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.active_joint_names
        ]
        self.active_qpos_idx = [self.model.jnt_qposadr[jid] for jid in self.active_joint_ids]
        self.active_qvel_idx = [self.model.jnt_dofadr[jid] for jid in self.active_joint_ids]

        self.active_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.active_actuator_names
        ]
        self.other_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.other_actuator_names
        ]
        self.active_gripper_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.active_gripper_actuator_names
        ]
        self.other_gripper_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.other_gripper_actuator_names
        ]

        self.finger_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"gripper_{self.arm}_right_finger_link"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"gripper_{self.arm}_left_finger_link"),
        ]

        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_lift_link"
        )
        self.wrist_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"arm_{self.arm}_7_link"
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

        self.right_home_pose = np.array([0.10, 1.00, 1.30, 1.55, -1.15, 1.00, 0.00], dtype=np.float64)
        self.left_home_pose = np.array([0.10, 1.00, 1.30, 1.55, -1.15, 1.00, 0.00], dtype=np.float64)

        self.gripper_open = 0.03

        self.ctrl = self.ctrl_home.copy()
        self._apply_compact_home_targets()

        self.finger_subtrees = [
            self._descendant_body_ids(self.finger_body_ids[0]),
            self._descendant_body_ids(self.finger_body_ids[1]),
        ]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        self.viewer = None
        self.goal = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.prev_dist = None

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
        right_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_right_{i}_joint_position")
            for i in range(1, 8)
        ]
        for i, aid in enumerate(right_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.ctrl_home[aid] = np.clip(self.right_home_pose[i], lo, hi)

        left_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_left_{i}_joint_position")
            for i in range(1, 8)
        ]
        for i, aid in enumerate(left_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.ctrl_home[aid] = np.clip(self.left_home_pose[i], lo, hi)

        all_grippers = self.active_gripper_actuator_ids + self.other_gripper_actuator_ids
        for aid in all_grippers:
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.ctrl_home[aid] = np.clip(self.gripper_open, lo, hi)

    def _hold_reference(self):
        self.data.qpos[self.reference_qpos_adr : self.reference_qpos_adr + 7] = self.reference_qpos0
        self.data.qvel[self.reference_dof_adr : self.reference_dof_adr + 6] = 0.0

    def _gripper_base_pos(self):
        return self.data.xpos[self.wrist_body_id].copy()

    def _gripper_mid_pos(self):
        p1 = self.data.xpos[self.finger_body_ids[0]].copy()
        p2 = self.data.xpos[self.finger_body_ids[1]].copy()
        return 0.5 * (p1 + p2)

    def _gripper_axis(self):
        base = self._gripper_base_pos()
        mid = self._gripper_mid_pos()
        axis = mid - base
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis / norm

    def _farthest_geom_point_on_subtree(self, subtree_body_ids):
        axis = self._gripper_axis()
        base = self._gripper_base_pos()

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
            return self._gripper_mid_pos().copy()

        return best_pos

    def _gripper_action_center_pos(self):
        tip_a = self._farthest_geom_point_on_subtree(self.finger_subtrees[0])
        tip_b = self._farthest_geom_point_on_subtree(self.finger_subtrees[1])
        return 0.5 * (tip_a + tip_b)

    def _ee_pos(self):
        return self._gripper_action_center_pos()

    def _geom_bounding_radius(self, gid):
        size = self.model.geom_size[gid]
        return float(np.max(size))

    def _goal_collides_with_robot(self, goal):
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            if bid <= 0:
                continue

            geom_pos = self.data.geom_xpos[gid]
            geom_r = self._geom_bounding_radius(gid)
            d = np.linalg.norm(goal - geom_pos)

            if d < (self.goal_radius + geom_r + self.goal_clearance):
                return True
        return False

    def _get_obs(self):
        q = np.array([self.data.qpos[i] for i in self.active_qpos_idx], dtype=np.float32)
        dq = np.array([self.data.qvel[i] for i in self.active_qvel_idx], dtype=np.float32)
        ee = self._ee_pos().astype(np.float32)
        goal = self.goal.astype(np.float32)
        err = goal - ee
        obs = np.concatenate([q, dq, ee, goal, err]).astype(np.float32)
        dist = float(np.linalg.norm(err))
        return obs, dist

    def _sample_goal(self):
        torso = self.data.xpos[self.torso_body_id].copy()
        ee0 = self._ee_pos().copy()
        base = self._gripper_base_pos().copy()

        # Centro simétrico respecto al robot
        center = torso + np.array([0.30, self.side_sign * 0.12, -0.14], dtype=np.float64)

        for _ in range(600):
            g = center + np.array([
                self.np_random.uniform(-0.16, 0.16),
                self.np_random.uniform(-0.12, 0.12),
                self.np_random.uniform(-0.12, 0.12),
            ])

            g[0] = max(g[0], torso[0] + 0.12)
            g[2] = np.clip(g[2], 0.28, 0.85)

            if self.arm == "right":
                g[1] = min(g[1], torso[1] - 0.02)
            else:
                g[1] = max(g[1], torso[1] + 0.02)

            dist_ee = np.linalg.norm(g - ee0)
            dist_base = np.linalg.norm(g - base)
            dist_torso = np.linalg.norm(g - torso)

            if not (0.10 <= dist_ee <= 0.34):
                continue
            if not (0.16 <= dist_base <= 0.40):
                continue
            if not (0.22 <= dist_torso <= 0.48):
                continue
            if self._goal_collides_with_robot(g):
                continue

            return g.astype(np.float64)

        # fallback simétrico, también corregido contra colisión
        fallback = center.copy()
        fallback[0] = max(fallback[0], torso[0] + 0.12)
        fallback[2] = np.clip(fallback[2], 0.28, 0.85)
        if self.arm == "right":
            fallback[1] = min(fallback[1], torso[1] - 0.02)
        else:
            fallback[1] = max(fallback[1], torso[1] + 0.02)

        if self._goal_collides_with_robot(fallback):
            # empujarlo hacia afuera del robot
            fallback = fallback + np.array([0.08, self.side_sign * 0.05, 0.05], dtype=np.float64)

        return fallback.astype(np.float64)

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

        for _ in range(140):
            self.data.ctrl[:] = self.ctrl
            mujoco.mj_step(self.model, self.data)
            self._hold_reference()
            mujoco.mj_forward(self.model, self.data)

        self.goal = self._sample_goal()
        obs, dist = self._get_obs()
        self.prev_dist = dist

        if self.render_mode == "human":
            self.render()

        return obs, {"goal": self.goal.copy(), "arm": self.arm}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -0.5, 0.5).astype(np.float64)

        current_targets = self.ctrl[self.active_actuator_ids].copy()
        delta = action * self.action_scale
        max_delta = 0.002
        delta = np.clip(delta, -max_delta, max_delta)
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
            "arm": self.arm,
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
                size=np.array([self.goal_radius, 0.0, 0.0]),
                pos=self.goal,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.1, 0.1, 1.0]),
            )

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    arm = "right"
    if len(sys.argv) > 1:
        arm = sys.argv[1]
    env = TiagoArmReachEnv(arm=arm, render_mode="human")
    obs, info = env.reset()
    print("Arm:", arm, "Goal:", info["goal"])
    try:
        while True:
            action = np.zeros(7, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
