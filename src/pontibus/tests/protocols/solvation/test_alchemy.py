# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import openfe
import openmm
import pytest
from openfe.tests.protocols.openmm_ahfe.test_ahfe_protocol import (
    _assert_num_forces,
)
from openff.interchange.interop.openmm import to_openmm_positions
from openff.toolkit import Molecule
from openmm import (
    CustomBondForce,
    CustomNonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    MonteCarloBarostat,
    NonbondedForce,
    PeriodicTorsionForce,
)
from openmm import unit as omm_unit
from openmmtools.alchemy import AlchemicalRegion
from openmmtools.tests.test_alchemy import (
    check_interacting_energy_components,
    check_noninteracting_energy_components,
    compare_system_energies,
    overlap_check,
)

from pontibus.components.extended_solvent_component import ExtendedSolventComponent
from pontibus.protocols.solvation.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)
from pontibus.utils.experimental_absolute_factory import (
    ExperimentalAbsoluteAlchemicalFactory,
)
from pontibus.utils.molecules import WATER
from pontibus.utils.system_creation import interchange_system_creation


def _nonbonded_force(system: openmm.System) -> openmm.NonbondedForce:
    forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    assert len(forces) == 1
    return forces[0]


@pytest.fixture(scope="module")
def vinyl_chloride():
    m = Molecule.from_smiles("C=CCl")
    m.generate_conformers(n_conformers=1)
    m.assign_partial_charges(partial_charge_method="gasteiger")
    return m


@pytest.fixture(scope="module")
def water_off():
    return WATER.to_openff()


class TestSoluteVSite:
    @pytest.fixture(scope="class")
    def interchange(self, vinyl_chloride, vsite_offxml, water_off):
        solute = openfe.SmallMoleculeComponent.from_openff(vinyl_chloride)
        interchange, comp_resids = interchange_system_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[vsite_offxml],
            ),
            solvation_settings=PackmolSolvationSettings(
                number_of_solvent_molecules=1000,
                solvent_padding=None,
            ),
            smc_components={solute: vinyl_chloride},
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange

    @pytest.fixture(scope="class")
    def omm_system(self, interchange):
        return interchange.to_openmm_system()

    @pytest.fixture(scope="class")
    def positions(self, interchange):
        return to_openmm_positions(interchange, include_virtual_sites=True)

    @pytest.fixture(scope="class")
    def alchemical_indices(self):
        # vinyl chloride is 6 atoms; 1000 waters * 3 sites = 3000 atoms
        # so the solute's single virtual site is particle index 3006.
        return [0, 1, 2, 3, 4, 5, 3006]

    @pytest.fixture(scope="class")
    def alchemical_region(self, alchemical_indices):
        return AlchemicalRegion(alchemical_atoms=alchemical_indices)

    @pytest.fixture(scope="class")
    def alchemical_system(self, omm_system, alchemical_region):
        alchemical_factory = ExperimentalAbsoluteAlchemicalFactory()
        return alchemical_factory.create_alchemical_system(omm_system, alchemical_region)

    def test_particle_count_preserved(self, omm_system, alchemical_system):
        assert alchemical_system.getNumParticles() == omm_system.getNumParticles()

    def test_virtual_sites_preserved(self, omm_system, alchemical_system):
        assert omm_system.isVirtualSite(3006)

        for i in range(omm_system.getNumParticles()):
            assert alchemical_system.isVirtualSite(i) == omm_system.isVirtualSite(i)

        ref = omm_system.getVirtualSite(3006)
        alch = alchemical_system.getVirtualSite(3006)
        assert type(alch) is type(ref)
        assert alch.getNumParticles() == ref.getNumParticles()
        assert [alch.getParticle(p) for p in range(alch.getNumParticles())] == [
            ref.getParticle(p) for p in range(ref.getNumParticles())
        ]

    def test_masses_and_constraints_preserved(self, omm_system, alchemical_system):
        assert alchemical_system.getNumConstraints() == omm_system.getNumConstraints()
        for i in range(omm_system.getNumParticles()):
            assert alchemical_system.getParticleMass(i) == omm_system.getParticleMass(i)

    def test_alchemical_atoms_zeroed_in_nonbonded_force(
        self, alchemical_system, alchemical_indices
    ):
        nbf = _nonbonded_force(alchemical_system)
        for i in alchemical_indices:
            charge, _, epsilon = nbf.getParticleParameters(i)
            assert charge.value_in_unit(omm_unit.elementary_charge) == pytest.approx(0.0)
            assert epsilon.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.0)

    def test_environment_atoms_untouched_in_nonbonded_force(
        self, omm_system, alchemical_system, alchemical_indices
    ):
        ref = _nonbonded_force(omm_system)
        alch = _nonbonded_force(alchemical_system)
        env = set(range(omm_system.getNumParticles())) - set(alchemical_indices)
        for i in sorted(env):
            assert alch.getParticleParameters(i) == ref.getParticleParameters(i)

    def test_electrostatics_offsets_match_reference_charges(
        self, omm_system, alchemical_system, alchemical_indices
    ):
        ref = _nonbonded_force(omm_system)
        alch = _nonbonded_force(alchemical_system)

        offsets = {}
        for k in range(alch.getNumParticleParameterOffsets()):
            name, particle, q_scale, s_scale, e_scale = alch.getParticleParameterOffset(k)
            assert name == "lambda_electrostatics"
            assert (s_scale, e_scale) == (0.0, 0.0)
            offsets[particle] = q_scale

        # one offset per alchemical atom, none for the environment
        assert set(offsets) == set(alchemical_indices)

        for i in alchemical_indices:
            ref_charge = ref.getParticleParameters(i)[0].value_in_unit(omm_unit.elementary_charge)
            assert offsets[i] == pytest.approx(ref_charge)

    def test_number_of_forces(self, alchemical_system):
        _assert_num_forces(alchemical_system, NonbondedForce, 1)
        _assert_num_forces(alchemical_system, CustomNonbondedForce, 2)
        _assert_num_forces(alchemical_system, CustomBondForce, 2)
        _assert_num_forces(alchemical_system, HarmonicBondForce, 1)
        _assert_num_forces(alchemical_system, HarmonicAngleForce, 1)
        _assert_num_forces(alchemical_system, PeriodicTorsionForce, 1)
        _assert_num_forces(alchemical_system, MonteCarloBarostat, 0)

    def test_compare_energies(self, omm_system, alchemical_system, alchemical_region, positions):
        compare_system_energies(omm_system, alchemical_system, alchemical_region, positions)

    def test_noninteracting_energies(
        self, omm_system, alchemical_system, alchemical_region, positions
    ):
        check_noninteracting_energy_components(
            omm_system,
            alchemical_system,
            alchemical_region,
            positions,
        )

    def test_interacting_energies(
        self, omm_system, alchemical_system, alchemical_region, positions
    ):
        check_interacting_energy_components(
            omm_system,
            alchemical_system,
            alchemical_region,
            positions,
        )

    @pytest.mark.slow  # pragma: no cover
    def test_overlap(self, omm_system, alchemical_system, positions):
        overlap_check(
            omm_system,
            alchemical_system,
            positions,
            cached_trajectory_filename=None,
            name="test",
        )
