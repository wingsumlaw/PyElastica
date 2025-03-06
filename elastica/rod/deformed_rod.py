__doc__ = """ Implementation of a deformed rod class that assumes all undeformed configurations are straight rods with even node spacing. """

import matplotlib.pyplot as plt

import numpy as np
from elastica.rod import RodBase
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)

from elastica.rod.knot_theory import KnotTheory
from elastica._calculus import (
    _difference,
    _average,
)

from elastica.rod.cosserat_rod import (
    _compute_internal_forces,
    _compute_internal_torques,
    _update_accelerations,
    _zeroed_out_external_forces_and_torques,
    _compute_shear_stretch_strains,
    _compute_bending_twist_strains,
)

from numpy.testing import assert_allclose
from elastica.utils import MaxDimension, Tolerance
from elastica.rod.factory_function import (
    _directors_validity_checker,
    _assert_dim,
    _assert_shape,
)

position_difference_kernel = _difference
position_average = _average


class DeformedStraightRod(RodBase, KnotTheory):
    """
    Deformed rod class that assumes all undeformed configurations are straight rods with even node spacing.

        Attributes
        ----------
        n_elems : int
            Number of elements in the rod
        position_collection : numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node position vectors.
            Unlike for the CosseratRod straight_rod, this is a required input for the DeformedStraightRod.
        velocity_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node velocity vectors.
        acceleration_collection: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node acceleration vectors.
        omega_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Array containing element angular velocity vectors.
        alpha_collection: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Array contining element angular acceleration vectors.
        director_collection: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Array containing element director matrices.
        rest_lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths at rest configuration.
        density: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod elements densities.
        volume: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element volumes.
        mass: numpy.ndarray
            1D (n_nodes) array containing data with 'float' type.
            Rod node masses. Note that masses are stored on the nodes, not on elements.
        mass_second_moment_of_inertia: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element mass second moment of interia.
        inv_mass_second_moment_of_inertia: numpy.ndarray
            3D (dim, dim, n_elems) array containing data with 'float' type.
            Rod element inverse mass moment of inertia.
        rest_voronoi_lengths: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod lengths on the voronoi domain at the rest configuration.
        internal_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Rod node internal forces. Note that internal forces are stored on the node, not on elements.
        internal_torques: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element internal torques.
        external_forces: numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            External forces acting on rod nodes.
        external_torques: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            External torques acting on rod elements.
        lengths: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element lengths.
        tangents: numpy.ndarray
            2D (dim, n_elems) array containing data with 'float' type.
            Rod element tangent vectors.
        radius: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element radius.
        dilatation: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation.
        voronoi_dilatation: numpy.ndarray
            1D (n_voronoi) array containing data with 'float' type.
            Rod dilatation on voronoi domain.
        dilatation_rate: numpy.ndarray
            1D (n_elems) array containing data with 'float' type.
            Rod element dilatation rates.
    """

    def __init__(
        self,
        n_elements,
        position,
        velocity,
        omega,
        acceleration,
        angular_acceleration,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density,
        volume,
        mass,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
        ring_rod_flag,
        **kwargs,
    ):
        self.n_elems = n_elements
        self.position_collection = position
        self.velocity_collection = velocity
        self.omega_collection = omega
        self.acceleration_collection = acceleration
        self.alpha_collection = angular_acceleration
        self.director_collection = directors
        self.radius = radius
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.shear_matrix = shear_matrix
        self.bend_matrix = bend_matrix
        self.density = density
        self.volume = volume
        self.mass = mass
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.rest_lengths = rest_lengths
        self.tangents = tangents
        self.dilatation = dilatation
        self.dilatation_rate = dilatation_rate
        self.voronoi_dilatation = voronoi_dilatation
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple
        self.ring_rod_flag = ring_rod_flag
        self.__dict__.update(kwargs)

        if not self.ring_rod_flag:
            # For ring rod there are no periodic elements so below code won't run.
            # We add periodic elements at the memory block construction.
            # Compute shear stretch and strains.
            _compute_shear_stretch_strains(
                self.position_collection,
                self.volume,
                self.lengths,
                self.tangents,
                self.radius,
                self.rest_lengths,
                self.rest_voronoi_lengths,
                self.dilatation,
                self.voronoi_dilatation,
                self.director_collection,
                self.sigma,
            )

            # Compute bending twist strains
            _compute_bending_twist_strains(
                self.director_collection, self.rest_voronoi_lengths, self.kappa
            )

    @classmethod
    def planar_deformed_rod(
        cls,
        n_elements: int,
        normal: np.ndarray,
        init_positions: np.ndarray,
        base_radius: float,
        base_length: float,
        density: float,
        youngs_modulus: float,
        **kwargs,
    ):
        """
        Create a new deformed rod object. Assumes it is planar with a normal defining the facing of the plane.

        Parameters
        ----------
        n_elements : int
            Number of elements in the rod
        direction : numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Rod direction vector.
        normal : numpy.ndarray
            1D (dim) array containing data with 'float' type.
            Rod normal vector.
        init_positions : numpy.ndarray
            2D (dim, n_nodes) array containing data with 'float' type.
            Array containing node position vectors.
        base_radius : float
            Radius of the rod.
        density : float
            Rod density.
        youngs_modulus : float
            Rod Young's modulus.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        DeformedStraightRod
            New deformed rod object.
        """

        ring_rod_flag = False

        # Get directions from initial positions
        direction = (init_positions[:, 1] - init_positions[:, 0]) / np.linalg.norm(
            init_positions[:, 1] - init_positions[:, 0]
        )

        (
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            ring_rod_flag,
        ) = _deformed_rod_allocate(
            base_length,
            direction,
            normal,
            n_elements,
            init_positions,
            base_radius,
            density,
            youngs_modulus,
        )

        return cls(
            n_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
            ring_rod_flag,
            base_radius=base_radius,
            base_length=base_length,
            youngs_modulus=youngs_modulus,
            init_positions=init_positions,
        )

    # ************************
    # COMPUTE ROD STATE AS IMPLEMENTED BY PYELASTICA
    # ************************

    def compute_internal_forces_and_torques(self, time):
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.

        Parameters
        ----------
        time: float
            current time

        """
        _compute_internal_forces(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
            self.rest_sigma,
            self.shear_matrix,
            self.internal_stress,
            self.internal_forces,
            self.ghost_elems_idx,
        )

        _compute_internal_torques(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.bend_matrix,
            self.rest_kappa,
            self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time):
        """
        Updates the acceleration variables

        Parameters
        ----------
        time: float
            current time

        """
        _update_accelerations(
            self.acceleration_collection,
            self.internal_forces,
            self.external_forces,
            self.mass,
            self.alpha_collection,
            self.inv_mass_second_moment_of_inertia,
            self.internal_torques,
            self.external_torques,
            self.dilatation,
        )

    def zeroed_out_external_forces_and_torques(self, time):
        _zeroed_out_external_forces_and_torques(
            self.external_forces, self.external_torques
        )

    def compute_translational_energy(self):
        """
        Compute total translational energy of the rod at the instance.
        """
        return (
            0.5
            * (
                self.mass
                * np.einsum(
                    "ij, ij-> j", self.velocity_collection, self.velocity_collection
                )
            ).sum()
        )

    def compute_rotational_energy(self):
        """
        Compute total rotational energy of the rod at the instance.
        """
        J_omega_upon_e = (
            _batch_matvec(self.mass_second_moment_of_inertia, self.omega_collection)
            / self.dilatation
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, J_omega_upon_e).sum()

    def compute_velocity_center_of_mass(self):
        """
        Compute velocity center of mass of the rod at the instance.
        """
        mass_times_velocity = np.einsum("j,ij->ij", self.mass, self.velocity_collection)
        sum_mass_times_velocity = np.einsum("ij->i", mass_times_velocity)

        return sum_mass_times_velocity / self.mass.sum()

    def compute_position_center_of_mass(self):
        """
        Compute position center of mass of the rod at the instance.
        """
        mass_times_position = np.einsum("j,ij->ij", self.mass, self.position_collection)
        sum_mass_times_position = np.einsum("ij->i", mass_times_position)

        return sum_mass_times_position / self.mass.sum()

    def compute_bending_energy(self):
        """
        Compute total bending energy of the rod at the instance.
        """

        kappa_diff = self.kappa - self.rest_kappa
        bending_internal_torques = _batch_matvec(self.bend_matrix, kappa_diff)

        return (
            0.5
            * (
                _batch_dot(kappa_diff, bending_internal_torques)
                * self.rest_voronoi_lengths
            ).sum()
        )

    def compute_shear_energy(self):
        """
        Compute total shear energy of the rod at the instance.
        """

        sigma_diff = self.sigma - self.rest_sigma
        shear_internal_forces = _batch_matvec(self.shear_matrix, sigma_diff)

        return (
            0.5
            * (_batch_dot(sigma_diff, shear_internal_forces) * self.rest_lengths).sum()
        )


def _deformed_rod_allocate(
    base_length: float,
    direction: np.ndarray,
    normal: np.ndarray,
    n_elements: int,
    init_positions: np.ndarray,
    base_radius: float,
    density: float,
    youngs_modulus: float,
    **kwargs,
):
    ring_rod_flag = False
    n_nodes = n_elements + 1
    # Sanity checks for length and initial direction
    assert base_length > Tolerance.atol()
    assert np.sqrt(np.dot(normal, normal)) > Tolerance.atol()
    assert np.sqrt(np.dot(direction, direction)) > Tolerance.atol()

    n_voronoi_elements = n_elements - 1

    # Assumes even lengths and undeformed when in straight line
    rest_lengths = np.array([base_length / n_elements for _ in range(n_elements)])
    rest_voronoi_lengths = np.array(
        [base_length / n_elements for _ in range(n_voronoi_elements)]
    )

    # If preexisting directors
    if "preexisting_directors" in kwargs:
        directors = kwargs["preexisting_directors"]
        # normals = directors[0, ...]
        # tangents = directors[2, ...]
        # assert_allclose(
        #     _batch_dot(normals, tangents),
        #     0,
        #     atol=Tolerance.atol(),
        #     err_msg="Rod tangents and normals are not perpendicular",
        # )
        # _directors_validity_checker(directors, tangents, n_elements)
    else:
        # Determine position differences
        position_diff = init_positions[..., 1:] - init_positions[..., :-1]
        position_norms = _batch_norm(position_diff)
        tangents = position_diff / position_norms
        normal /= np.linalg.norm(normal)

        # Determine directors
        directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
        normals = np.repeat(normal[:, np.newaxis], n_elements, axis=1)

        # Make sure rod normal and tangent are perpendicular
        assert_allclose(
            _batch_dot(normals, tangents),
            0,
            atol=Tolerance.atol(),
            err_msg="Rod tangents and normals are not perpendicular",
        )
        directors[0, ...] = normals
        directors[1, ...] = _batch_cross(tangents, normals)
        directors[2, ...] = tangents
        _directors_validity_checker(directors, tangents, n_elements)

    # Assume constant radius
    radius = np.zeros((n_elements))
    radius_temp = np.array(base_radius)
    _assert_dim(radius_temp, 2, "radius")
    radius[:] = radius_temp
    assert np.all(radius > Tolerance.atol()), " Radius has to be greater than 0."

    # Make ends heavier
    # Set density array using regular lengths
    density_array = np.zeros((n_elements))
    density_temp = np.array(density)
    _assert_dim(density_temp, 2, "density")
    density_array[:] = density_temp
    density_array[0] = density * 1000
    density_array[-1] = density * 1000

    assert np.all(
        density_array > Tolerance.atol()
    ), " Density has to be greater than 0."

    # Second moment of inertia
    A0 = np.pi * radius * radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3]).transpose()
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    mass_second_moment_of_inertia_temp = np.einsum(
        "ij,i->ij", I0, density * rest_lengths
    )

    for i in range(n_elements):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )

    # Inverse of second moment of inertia
    inv_mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements)
    )
    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert (
            np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
            == MaxDimension.value()
        )
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )

    # Shear/stretch matrix
    shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))

    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    shear_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            shear_matrix[..., i],
            [
                alpha_c * shear_modulus * A0[i],
                alpha_c * shear_modulus * A0[i],
                youngs_modulus * A0[i],
            ],
        )

    # Bend/Twist matrix
    bend_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_voronoi_elements + 1), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            bend_matrix[..., i],
            [
                youngs_modulus * I0_1[i],
                youngs_modulus * I0_2[i],
                shear_modulus * I0_3[i],
            ],
        )

    for i in range(0, MaxDimension.value()):
        assert np.all(
            bend_matrix[i, i, :] > Tolerance.atol()
        ), " Bend matrix has to be greater than 0."

    # Compute bend matrix in Voronoi Domain
    bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths[1:]
        + bend_matrix[..., :-1] * rest_lengths[0:-1]
    ) / (rest_lengths[1:] + rest_lengths[:-1])

    # Compute volume of elements
    volume = np.pi * radius ** 2 * rest_lengths

    # Compute mass of elements
    mass = np.zeros(n_nodes)
    mass[:-1] += 0.5 * density * volume
    mass[1:] += 0.5 * density * volume

    # set rest strains and curvature to be  zero at start
    rest_sigma = np.zeros((MaxDimension.value(), n_elements))
    _assert_shape(rest_sigma, (MaxDimension.value(), n_elements), "rest_sigma")

    rest_kappa = np.zeros((MaxDimension.value(), n_voronoi_elements))
    _assert_shape(rest_kappa, (MaxDimension.value(), n_voronoi_elements), "rest_kappa")

    # Allocate arrays for Cosserat Rod equations
    velocities = np.zeros((MaxDimension.value(), n_nodes))
    omegas = np.zeros((MaxDimension.value(), n_elements))
    accelerations = 0.0 * velocities
    angular_accelerations = 0.0 * omegas

    internal_forces = 0.0 * accelerations
    internal_torques = 0.0 * angular_accelerations

    external_forces = 0.0 * accelerations
    external_torques = 0.0 * angular_accelerations

    lengths = np.zeros((n_elements))
    tangents = np.zeros((3, n_elements))

    dilatation = np.zeros((n_elements))
    voronoi_dilatation = np.zeros((n_voronoi_elements))
    dilatation_rate = np.zeros((n_elements))

    sigma = np.zeros((3, n_elements))
    kappa = np.zeros((3, n_voronoi_elements))

    internal_stress = np.zeros((3, n_elements))
    internal_couple = np.zeros((3, n_voronoi_elements))

    position = init_positions

    return (
        n_elements,
        position,
        velocities,
        omegas,
        accelerations,
        angular_accelerations,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density_array,
        volume,
        mass,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
        ring_rod_flag,
    )
