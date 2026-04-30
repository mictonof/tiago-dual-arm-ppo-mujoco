import mujoco
from tiago_right_arm_reach_env import TiagoRightArmReachEnv

env = TiagoRightArmReachEnv(render_mode=None)
env.reset(seed=1)

def bname(i):
    return mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i)

def gname(i):
    return mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)

def descendants(root_id):
    parent = env.model.body_parentid
    out = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for bid in range(env.model.nbody):
            if int(parent[bid]) in out and bid not in out:
                out.add(int(bid))
                changed = True
    return sorted(out)

right_finger_ids = env.right_finger_body_ids
wrist_id = env.right_wrist_body_id

print("\n=== WRIST ===")
print(wrist_id, bname(wrist_id), env.data.xpos[wrist_id])

for idx, fid in enumerate(right_finger_ids):
    print(f"\n=== FINGER {idx} ROOT ===")
    print(fid, bname(fid), env.data.xpos[fid])

    sub = descendants(fid)
    print("Bodies in subtree:")
    for bid in sub:
        print(" ", bid, bname(bid), env.data.xpos[bid])

    print("Geoms in subtree:")
    for gid in range(env.model.ngeom):
        gb = int(env.model.geom_bodyid[gid])
        if gb in sub:
            print(" ", gid, gname(gid), "body=", bname(gb), "geom_xpos=", env.data.geom_xpos[gid])

env.close()
