# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import logging

import numpy.typing as npt
import openmm
from gufe import (
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from gufe.settings import SettingsBaseModel
from openfe.protocols.openmm_afe.base_afe_units import BaseAbsoluteSetupUnit
from openfe.protocols.openmm_afe.equil_afe_settings import AlchemicalSettings
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
)
from openfe.utils import without_oechem_backend
from openff.interchange.interop.openmm import to_openmm_positions
from openff.toolkit import Molecule as OFFMolecule
from openff.units.openmm import to_openmm
from openmm import app
from openmmtools.alchemy import (
    AbsoluteAlchemicalFactory,
    AlchemicalRegion,
)

from pontibus.utils.experimental_absolute_factory import (
    ExperimentalAbsoluteAlchemicalFactory,
)
from pontibus.utils.protocol_utils import _get_and_charge_solvent_offmol
from pontibus.utils.system_creation import interchange_system_creation
from pontibus.utils.system_manipulation import adjust_system

logger = logging.getLogger(__name__)


class BaseASFESetupUnit(BaseAbsoluteSetupUnit):
    simtype: str

    @staticmethod
    def _validate_vsites(system: openmm.System, integrator_settings: IntegratorSettings) -> None:
        """
        Validate virtual site handling for alchemical system.

        Parameters
        ----------
        System : openmm.System
          System to validate.
        integrator_settings : IntegratorSettings
          Langevin integrator settings to verify against.

        Returns
        -------
        None

        Notes
        -----
        * Small placeholder for a larger thing.
        """
        if not integrator_settings.reassign_velocities:
            has_vsite = any(system.isVirtualSite(i) for i in range(system.getNumParticles()))
            if has_vsite:
                errmsg = "Simulations with virtual sites without velocity reassignment are unstable"
                raise ValueError(errmsg)

    def _get_omm_objects(
        self,
        settings: dict[str, SettingsBaseModel],
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> tuple[app.Topology, openmm.System, openmm.unit.Quantity, dict[str, npt.NDArray]]:
        """
        Get the OpenMM Topology, Positions and System of the
        parameterised system.

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          Protocol settings
        protein_component : Optional[ProteinComponent]
          Protein component for the system.
        solvent_component : Optional[SolventComponent]
          Solvent component for the system.
        small_mols : dict[SmallMoleculeComponent, OFFMolecule]
          SmallMoleculeComponents defining ligands to be added to the system

        Returns
        -------
        topology : app.Topology
          OpenMM Topology object describing the parameterized system.
        system : openmm.System
          An non-alchemical OpenMM System of the simulated system.
        positions : openmm.unit.Quantity
          Positions of the system.
        comp_resids : dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.

        Notes
        -----
        * For now this method solely calls interchange system creation for
          solvation.
        """
        if self.verbose:
            self.logger.info("Parameterizing system")

        # Set partial charges for all smcs
        self._assign_partial_charges(settings["charge_settings"], small_mols)

        # Get solvent offmol if necessary
        if solvent_component is not None:
            solvent_offmol = _get_and_charge_solvent_offmol(
                solvent_component,
                settings["solvation_settings"],
                settings["charge_settings"],
            )
        else:
            solvent_offmol = None

        # Create your interchange object
        with without_oechem_backend():
            interchange, comp_resids = interchange_system_creation(
                ffsettings=settings["forcefield_settings"],
                solvation_settings=settings["solvation_settings"],
                smc_components=small_mols,
                protein_component=protein_component,
                solvent_component=solvent_component,
                solvent_offmol=solvent_offmol,
            )

        # Get omm objects back
        omm_topology = interchange.to_openmm_topology(collate=False)
        omm_system = interchange.to_openmm_system(
            hydrogen_mass=settings["forcefield_settings"].hydrogen_mass
        )

        # Add a barostat if needed
        if solvent_component is not None:
            barostat = openmm.MonteCarloBarostat(
                to_openmm(settings["thermo_settings"].pressure),
                to_openmm(settings["thermo_settings"].temperature),
                settings["integrator_settings"].barostat_frequency.m,
            )
        else:
            barostat = None

        adjust_system(
            system=omm_system, remove_force_types=openmm.CMMotionRemover, add_forces=barostat
        )

        positions = to_openmm_positions(interchange, include_virtual_sites=True)

        # Post creation system validation
        self._validate_vsites(omm_system, settings["integrator_settings"])

        return omm_topology, omm_system, positions, comp_resids

    def _get_experimental_alchemical_system(
        self,
        topology: app.Topology,
        system: openmm.System,
        comp_resids: dict[Component, npt.NDArray],
        alchem_comps: dict[str, list[Component]],
        alchemical_settings: AlchemicalSettings,
    ) -> tuple[ExperimentalAbsoluteAlchemicalFactory, openmm.System, list[int]]:
        """
        Get an alchemically modified system and its associated factory using
        the ExperimentalAbsoluteAlchemicalFactory,

        Parameters
        ----------
        topology : openmm.Topology
          Topology of OpenMM System.
        system : openmm.System
          System to alchemically modify.
        comp_resids : dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.
        alchem_comps : dict[str, list[Component]]
          A dictionary of alchemical components for each end state.
        alchemical_settings : AlchemicalSettings
          Settings controlling how the alchemical system is built.

        Returns
        -------
        alchemical_factory : AbsoluteAlchemicalFactory
          Factory for creating an alchemically modified system.
        alchemical_system : openmm.System
          Alchemically modified system
        alchemical_indices : list[int]
          A list of atom indices for the alchemically modified
          species in the system.

        Notes
        -----
        In theory, this option should allow for virtual sites.

        TODO
        ----
        * Add support for all alchemical factory options
        """
        alchemical_indices = self._get_alchemical_indices(topology, comp_resids, alchem_comps)

        alchemical_region = AlchemicalRegion(
            alchemical_atoms=alchemical_indices,
            softcore_alpha=alchemical_settings.softcore_alpha,
            annihilate_electrostatics=True,
            annihilate_sterics=alchemical_settings.annihilate_sterics,
            softcore_a=alchemical_settings.softcore_a,
            softcore_b=alchemical_settings.softcore_b,
            softcore_c=alchemical_settings.softcore_c,
            softcore_beta=0.0,
            softcore_d=1.0,
            softcore_e=1.0,
            softcore_f=2.0,
        )

        alchemical_factory = ExperimentalAbsoluteAlchemicalFactory(
            consistent_exceptions=False,
            switch_width=1.0 * openmm.unit.angstroms,
            alchemical_pme_treatment="exact",
            alchemical_rf_treatment="switched",
            disable_alchemical_dispersion_correction=alchemical_settings.disable_alchemical_dispersion_correction,
            split_alchemical_forces=True,
        )
        alchemical_system = alchemical_factory.create_alchemical_system(system, alchemical_region)

        return alchemical_factory, alchemical_system, alchemical_indices

    def _get_alchemical_system(
        self,
        topology: app.Topology,
        system: openmm.System,
        comp_resids: dict[Component, npt.NDArray],
        alchem_comps: dict[str, list[Component]],
        alchemical_settings: AlchemicalSettings,
    ) -> tuple[AbsoluteAlchemicalFactory, openmm.System, list[int]]:
        """
        Get an alchemically modified system and its associated factory.

        If the experimental settings are turned on, will return an
        ExperimentalAlchemicalFactory via
        :meth:`_get_experimental_alchemical_system`.

        Parameters
        ----------
        topology : openmm.Topology
          Topology of OpenMM System.
        system : openmm.System
          System to alchemically modify.
        comp_resids : dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.
        alchem_comps : dict[str, list[Component]]
          A dictionary of alchemical components for each end state.

        Returns
        -------
        alchemical_factory : AbsoluteAlchemicalFactory
          Factory for creating an alchemically modified system.
        alchemical_system : openmm.System
          Alchemically modified system
        alchemical_indices : list[int]
          A list of atom indices for the alchemically modified
          species in the system.
        alchemical_settings : AlchemicalSettings
          Settings controlling how the alchemical system is built.
        """
        if self._inputs["protocol"].settings.alchemical_settings.experimental:
            return self._get_experimental_alchemical_system(
                topology,
                system,
                comp_resids,
                alchem_comps,
                alchemical_settings,
            )
        else:
            return super()._get_alchemical_system(
                topology,
                system,
                comp_resids,
                alchem_comps,
                alchemical_settings,
            )
