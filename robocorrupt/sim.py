"""PyBullet reaching environment: 2-link planar arm, top-down camera, image observations.

The arm kinematics are analytic (no contact dynamics needed — this benchmark is
about *visual* robustness, not physics). PyBullet is used purely as a fast CPU
renderer. Each step the link bodies are teleported to the analytic FK pose.
"""
import math
import numpy as np

import pybullet as p


L1 = 0.30
L2 = 0.25
LINK_W = 0.05
ARM_H = 0.30          # height of the arm plane above the ground
RENDER_SIZE = 64
MAX_STEPS = 60
SUCCESS_DIST = 0.035
DQ_MAX = 0.12         # rad per step at |action| = 1
WORK_R_MIN = 0.15
WORK_R_MAX = 0.42
TGT_R = 0.05


def fk(q):
    """End-effector (x, y) for joint angles q = (q1, q2)."""
    q1, q2 = q
    x = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    y = L1 * math.sin(q1) + L2 * math.sin(q1 + q2)
    return np.array([x, y], dtype=np.float64)


def ik(xy):
    """Analytic 2R IK, elbow-down branch. Returns (q1, q2)."""
    x, y = float(xy[0]), float(xy[1])
    d2 = x * x + y * y
    cos_q2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    q2 = -math.acos(cos_q2)  # elbow down
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2),
                                       L1 + L2 * math.cos(q2))
    return np.array([q1, q2], dtype=np.float64)


def _yaw_quat(yaw):
    return p.getQuaternionFromEuler([0, 0, yaw])


class ReachingEnv:
    def __init__(self, render_size=RENDER_SIZE, max_steps=MAX_STEPS, seed=0):
        self.render_size = render_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self._cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(
            __import__("pybullet_data").getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self._cid)
        # -- link bodies (floating boxes, teleported each step) --
        blue = [0.15, 0.35, 0.85, 1.0]
        dark = [0.08, 0.18, 0.45, 1.0]
        self._link1 = self._make_box([L1, LINK_W, LINK_W], blue)
        self._link2 = self._make_box([L2, LINK_W, LINK_W], blue)
        self._j0 = self._make_sphere(0.034, dark)
        self._j1 = self._make_sphere(0.030, dark)
        self._ee = self._make_sphere(0.024, [0.1, 0.8, 0.2, 1.0])
        self._tgt = self._make_cylinder(TGT_R, 0.03, [0.9, 0.15, 0.15, 1.0])
        # -- top-down camera (zoomed: workspace fills the frame) --
        self._view = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 1.0],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0],
            physicsClientId=self._cid)
        self._proj = p.computeProjectionMatrixFOV(
            fov=50, aspect=1.0, nearVal=0.1, farVal=3.0,
            physicsClientId=self._cid)
        self.q = np.zeros(2)
        self.target = np.zeros(2)
        self.t = 0

    def _make_box(self, half, rgba):
        c = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                   physicsClientId=self._cid)
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba,
                                physicsClientId=self._cid)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                 baseVisualShapeIndex=v,
                                 basePosition=[0, 0, -10],
                                 physicsClientId=self._cid)

    def _make_sphere(self, r, rgba):
        c = p.createCollisionShape(p.GEOM_SPHERE, radius=r,
                                   physicsClientId=self._cid)
        v = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgba,
                                physicsClientId=self._cid)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                 baseVisualShapeIndex=v,
                                 basePosition=[0, 0, -10],
                                 physicsClientId=self._cid)

    def _make_cylinder(self, r, h, rgba):
        c = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                   physicsClientId=self._cid)
        v = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                rgbaColor=rgba, physicsClientId=self._cid)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                 baseVisualShapeIndex=v,
                                 basePosition=[0, 0, -10],
                                 physicsClientId=self._cid)

    def _place(self, body, xy, yaw=0.0, z=ARM_H):
        p.resetBasePositionAndOrientation(
            body, [float(xy[0]), float(xy[1]), z], _yaw_quat(yaw),
            physicsClientId=self._cid)

    def _pose_arm(self):
        q1, q2 = self.q
        j0 = np.zeros(2)
        j1 = j0 + L1 * np.array([math.cos(q1), math.sin(q1)])
        ee = fk(self.q)
        self._place(self._j0, j0)
        self._place(self._link1, j0 + 0.5 * L1 * np.array(
            [math.cos(q1), math.sin(q1)]), yaw=q1)
        self._place(self._j1, j1)
        self._place(self._link2, j1 + 0.5 * L2 * np.array(
            [math.cos(q1 + q2), math.sin(q1 + q2)]), yaw=q1 + q2)
        self._place(self._ee, ee)
        self._place(self._tgt, self.target, z=0.01)

    def reset(self):
        ang = self.rng.uniform(-math.pi, math.pi)
        rad = self.rng.uniform(WORK_R_MIN, WORK_R_MAX)
        self.target = np.array([rad * math.cos(ang), rad * math.sin(ang)])
        self.q = self.rng.uniform(-math.pi, math.pi, size=2)
        self.t = 0
        self._pose_arm()
        return self.render()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.q = self.q + action * DQ_MAX
        # wrap to [-pi, pi]
        self.q = (self.q + math.pi) % (2 * math.pi) - math.pi
        self.t += 1
        self._pose_arm()
        obs = self.render()
        dist = float(np.linalg.norm(fk(self.q) - self.target))
        success = dist < SUCCESS_DIST
        done = success or self.t >= self.max_steps
        return obs, success, done

    def step_target(self, q_star):
        """Low-level tracking: move joints toward predicted target angles."""
        dq = np.asarray(q_star, dtype=np.float64) - self.q
        dq = (dq + math.pi) % (2 * math.pi) - math.pi
        return self.step(np.clip(dq / DQ_MAX, -1, 1))

    def render(self):
        w, h, rgb, _, _ = p.getCameraImage(
            self.render_size, self.render_size, self._view, self._proj,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self._cid)
        img = np.asarray(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return img

    def expert_action(self, noise=0.0):
        q_star = ik(self.target)
        dq = q_star - self.q
        dq = (dq + math.pi) % (2 * math.pi) - math.pi
        a = np.clip(dq / DQ_MAX, -1, 1)
        if noise > 0:
            a = np.clip(a + self.rng.normal(0, noise, size=2), -1, 1)
        return a.astype(np.float64)

    def close(self):
        p.disconnect(physicsClientId=self._cid)
