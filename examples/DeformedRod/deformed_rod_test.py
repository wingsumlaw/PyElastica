import numpy as np
import matplotlib.pyplot as plt

from elastica import *


class DeformedRodSimulator(
    BaseSystemCollection, Constraints, Forcing, Damping, CallBacks
):
    pass


class CallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["internal_forces"].append(
                system.internal_forces.copy()
            )
            self.callback_params["internal_torques"].append(
                system.internal_torques.copy()
            )
            self.callback_params["bending_energy"].append(
                system.compute_bending_energy()
            )
            self.callback_params["shear_energy"].append(system.compute_shear_energy())
            return


deformed_rod_sim = DeformedRodSimulator()
final_time = 1
damping_constant = 0.3
time_step = 1e-6
total_steps = int(final_time / time_step)
step_skip = 100

n_elem = 20
base_length = 6.32
base_radius = 0.1
youngs = 3e9
density = 100

x = np.linspace(-3.16, 3.16, n_elem + 1)
y = np.array([-0.1 - 0.01 * xi ** 2 for xi in x])
z = np.zeros_like(x)

init_positions = np.row_stack((x, y, z))
print(init_positions)

def_rod = DeformedStraightRod.planar_deformed_rod(
    n_elem,
    np.array([0.0, 0.0, 1.0]),
    init_positions,
    base_radius,
    base_length,
    density,
    youngs,
)

deformed_rod_sim.append(def_rod)

# add damping
deformed_rod_sim.dampen(def_rod).using(
    AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=time_step,
)

# Add gravity
deformed_rod_sim.add_forcing_to(def_rod).using(
    GravityForces, acc_gravity=-9.80665 * np.array([0.0, 1.0, 0.0])
)

# fix rod ends
deformed_rod_sim.constrain(def_rod).using(
    FixedConstraint, constrained_position_idx=(0, -1), constrained_director_idx=(0, -1)
)

callback_params = {
    "time": [],
    "step": [],
    "position": [],
    "directors": [],
    "internal_forces": [],
    "internal_torques": [],
    "bending_energy": [],
    "shear_energy": [],
}

deformed_rod_sim.collect_diagnostics(def_rod).using(
    CallBack, step_skip=step_skip, callback_params=callback_params
)

deformed_rod_sim.finalize()

timestepper = PositionVerlet()

integrate(timestepper, deformed_rod_sim, final_time, total_steps)

last_timestep_positions = callback_params["position"][-1]

x_positions = last_timestep_positions[0]
y_positions = last_timestep_positions[1]

plt.plot(x, y)
plt.plot(x_positions, y_positions)
plt.show(block=True)
