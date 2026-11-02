"""MuJoCo model of the ArmPi Mini, generated from the measured link lengths in kinematics.py.

Joint conventions match calibration.yaml / kinematics.py, so MuJoCo qpos (radians) is the same
angle the rest of the stack uses (degrees):
    base      yaw about +z, 0 = arm along +x
    shoulder  0 = upper arm horizontal (+x), 90 = straight up
    elbow     relative to upper arm, negative bends down
    wrist     relative to forearm
    gripper   two slide fingers, 0 = closed .. 1 = fully open (mapped in MujocoBackend)

Masses, servo torque and gains start as ESTIMATES (hobby PWM servos, ~0.2-0.3 N*m). They only
matter for the dynamics (sag, lag); kinematics are exact to the measurements. `python -m
armpi.sysid fit` fits them to filmed step moves of the real arm and writes sim_params.yaml,
which overrides the defaults below whenever it exists.
"""
import copy
from pathlib import Path

import yaml

from .kinematics import LINKS

# Per-joint servo model. kp: position gain (N*m/rad), torque: force limit (N*m),
# damping (N*m*s/rad), armature: reflected rotor inertia (kg*m^2).
SERVO = {
    "base": {"kp": 3.0, "torque": 0.25, "damping": 0.02, "armature": 0.001},
    "shoulder": {"kp": 3.0, "torque": 0.25, "damping": 0.02, "armature": 0.001},
    "elbow": {"kp": 3.0, "torque": 0.25, "damping": 0.02, "armature": 0.001},
    "wrist": {"kp": 2.0, "torque": 0.2, "damping": 0.01, "armature": 0.001},
}
# kg, per link (servos included). Guesses; weigh the parts if you can.
MASSES = {"upper_arm": 0.035, "forearm": 0.035, "hand": 0.03, "finger": 0.005}
GRIPPER_OPEN_WIDTH = 0.040  # m, total finger gap when fully open (placeholder, measure it)
PARAMS_PATH = Path(__file__).with_name("sim_params.yaml")


def load_params(path=PARAMS_PATH):
    """Defaults merged with sim_params.yaml (if present). Returns {"servo": .., "masses": ..}."""
    out = {"servo": copy.deepcopy(SERVO), "masses": dict(MASSES)}
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return out
    for j, vals in (data.get("servo") or {}).items():
        out["servo"].setdefault(j, {}).update(vals)
    out["masses"].update(data.get("masses") or {})
    return out


def build_mjcf(links=LINKS, servo=None, open_width=GRIPPER_OPEN_WIDTH, masses=None):
    """Return the MJCF xml string. All geometry in meters. servo/masses default to
    load_params() (defaults + sim_params.yaml); partial dicts are merged over that."""
    p = load_params()
    for j, vals in (servo or {}).items():
        p["servo"].setdefault(j, {}).update(vals)
    p["masses"].update(masses or {})
    servo, ms = p["servo"], p["masses"]
    m = {k: v / 1000.0 for k, v in links.items()}
    sh_z = m["turntable_to_shoulder"]
    ua, fa = m["upper_arm"], m["forearm"]
    wg, fl = m["wrist_to_gripper"], m["fingers"]
    half = open_width / 2

    def joint(name, lo, hi):
        s = servo[name]
        return (f'<joint name="{name}" type="hinge" axis="0 -1 0" range="{lo} {hi}" '
                f'damping="{s["damping"]}" armature="{s.get("armature", 0.001)}"/>')

    def act(name, lo, hi):  # ctrl is in radians (qpos units); no ctrlrange, limits live in Arm
        s = servo[name]
        return (f'<position name="{name}" joint="{name}" kp="{s["kp"]}" '
                f'forcerange="-{s["torque"]} {s["torque"]}"/>')

    return f"""
<mujoco model="armpi_mini">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".75 .75 .75"
             width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
    <material name="blue" rgba=".15 .55 .75 1"/>
    <material name="dark" rgba=".15 .15 .18 1"/>
  </asset>
  <worldbody>
    <light pos="0 -0.3 0.6" dir="0 0.4 -1"/>
    <geom name="table" type="plane" size="0.4 0.4 0.01" material="grid"/>
    <geom name="base_box" type="box" pos="0 0 {m['base_height']/2}"
          size="0.045 0.045 {m['base_height']/2}" material="blue" mass="0.4"/>
    <body name="turntable" pos="0 0 {m['base_height']}">
      <joint name="base" type="hinge" axis="0 0 1" range="-120 120"
             damping="{servo['base']['damping']}" armature="{servo['base'].get('armature', 0.001)}"/>
      <geom type="cylinder" size="0.03 0.006" pos="0 0 0.006" material="dark" mass="0.03"/>
      <geom type="box" size="0.012 0.02 {sh_z/2}" pos="0 0 {sh_z/2}" material="dark" mass="0.04"/>
      <body name="upper_arm" pos="{m['shoulder_offset']} 0 {sh_z}">
        {joint('shoulder', -10, 190)}
        <geom type="capsule" fromto="0 0 0 {ua} 0 0" size="0.008" material="blue" mass="{ms['upper_arm']}"/>
        <body name="forearm" pos="{ua} 0 0">
          {joint('elbow', -150, 150)}
          <geom type="capsule" fromto="0 0 0 {fa} 0 0" size="0.007" material="blue" mass="{ms['forearm']}"/>
          <body name="hand" pos="{fa} 0 0">
            {joint('wrist', -150, 150)}
            <geom type="box" size="{wg/2} 0.015 0.012" pos="{wg/2} 0 0" material="dark" mass="{ms['hand']}"/>
            <site name="wrist_cam" pos="{wg/2} 0 0.02" size="0.005"/>
            <body name="finger_l" pos="{wg} 0 0">
              <joint name="finger_l" type="slide" axis="0 1 0" range="0 {half}" damping="0.5"/>
              <geom type="box" size="{fl/2} 0.003 0.008" pos="{fl/2} 0.003 0" material="blue"
                    mass="{ms['finger']}" friction="1.5 0.02 0.001"/>
            </body>
            <body name="finger_r" pos="{wg} 0 0">
              <joint name="finger_r" type="slide" axis="0 -1 0" range="0 {half}" damping="0.5"/>
              <geom type="box" size="{fl/2} 0.003 0.008" pos="{fl/2} -0.003 0" material="blue"
                    mass="{ms['finger']}" friction="1.5 0.02 0.001"/>
            </body>
            <site name="fingertip" pos="{wg + fl} 0 0" size="0.004" rgba="1 0 0 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <!-- the turntable sits flush on the base box: without this, contact friction between
         them drags the base joint (the old "base lags ~5 deg" in Arm Studio) -->
    <exclude body1="world" body2="turntable"/>
    <exclude body1="finger_l" body2="finger_r"/>
  </contact>
  <equality>
    <joint joint1="finger_l" joint2="finger_r"/>
  </equality>
  <actuator>
    <position name="base" joint="base" kp="{servo['base']['kp']}"
              forcerange="-{servo['base']['torque']} {servo['base']['torque']}"/>
    {act('shoulder', -10, 190)}
    {act('elbow', -150, 150)}
    {act('wrist', -150, 150)}
    <position name="gripper" joint="finger_l" kp="20" ctrlrange="0 {half}" forcerange="-2 2"/>
  </actuator>
</mujoco>
"""


def load_model(**kw):
    import mujoco
    return mujoco.MjModel.from_xml_string(build_mjcf(**kw))
