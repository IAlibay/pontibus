#!/usr/bin/python

"""
A vendored and modified version of OpenMMTools' AbsoluteAlchemicalFactory.

The modifications here, although experimental, may support virtual sites.
"""

import copy
import logging

from openmmtools import forcefactories, utils
from openmmtools.alchemy.alchemy import (
    AbsoluteAlchemicalFactory,
    AlchemicalRegion,
)

logger = logging.getLogger(__name__)


class ExperimentalAbsoluteAlchemicalFactory(AbsoluteAlchemicalFactory):
    def create_alchemical_system(
        self,
        reference_system,
        alchemical_regions,
        alchemical_regions_interactions=frozenset(),
    ):
        """Create an alchemically modified version of the reference system.

        To alter the alchemical state of the returned system use AlchemicalState.

        Parameters
        ----------
        reference_system : openmm.System
            The system to use as a reference for the creation of the
            alchemical system. This will not be modified.
        alchemical_regions : AlchemicalRegion
            The region of the reference system to alchemically soften.
        alchemical_regions_interactions : Set[Tuple[int, int]], optional
            Set of alchemical region index pairs for interacting regions.
            By default, all alchemical regions interact only with the
            non-alchemical environment.

        Returns
        -------
        alchemical_system : openmm.System
            Alchemically-modified version of reference_system.

        """
        if alchemical_regions_interactions != frozenset():
            raise NotImplementedError("Interactions between alchemical regions is untested")

        logger.debug(
            f"Dictionary of interacting alchemical regions: {alchemical_regions_interactions}"
        )
        if isinstance(alchemical_regions, AlchemicalRegion):
            alchemical_regions = [alchemical_regions]
        logger.debug(f"Using {len(alchemical_regions)} alchemical regions")

        # Resolve alchemical regions.
        alchemical_regions = [
            self._resolve_alchemical_region(reference_system, alchemical_region)
            for alchemical_region in alchemical_regions
        ]

        # Check for duplicate alchemical atoms/bonds/angles/torsions.
        all_alchemical_elements = {
            element_type: set() for element_type in ["atoms", "bonds", "angles", "torsions"]
        }

        for alchemical_region in alchemical_regions:
            for element_type, all_elements in all_alchemical_elements.items():
                # Ignore None alchemical elements.
                region_elements = getattr(alchemical_region, "alchemical_" + element_type)
                if region_elements is None:
                    continue

                # Check if there are duplicates with previous regions.
                duplicate_elements = all_elements & region_elements
                if len(duplicate_elements) > 0:
                    raise ValueError(f"Two regions have duplicate {element_type}.")

                # Update all alchemical elements.
                all_alchemical_elements[element_type].update(region_elements)

        # Check for duplicate names
        alchemical_region_names = {
            alchemical_region.name for alchemical_region in alchemical_regions
        }
        if len(alchemical_region_names) != len(alchemical_regions):
            raise ValueError("Two alchemical regions have the same name")

        # Record timing statistics.
        timer = utils.Timer()
        timer.start("Create alchemically modified system")

        # Build alchemical system to modify. This copies particles, vsites,
        # constraints, box vectors and all the forces. We'll later remove
        # the forces that we remodel to be alchemically modified.
        alchemical_system = copy.deepcopy(reference_system)

        # Modify forces as appropriate. We delete the forces that
        # have been processed modified at the end of the for loop.
        forces_to_remove = []
        alchemical_forces_by_lambda = {}
        for force_index, reference_force in enumerate(reference_system.getForces()):
            # TODO switch to functools.singledispatch when we drop Python2 support
            reference_force_name = reference_force.__class__.__name__
            alchemical_force_creator_name = f"_alchemically_modify_{reference_force_name}"
            try:
                alchemical_force_creator_func = getattr(self, alchemical_force_creator_name)
            except AttributeError:
                pass
            else:
                # The reference system force will be deleted.
                forces_to_remove.append(force_index)
                # Collect all the Force objects modeling the reference force.
                alchemical_forces = alchemical_force_creator_func(
                    reference_force, alchemical_regions, alchemical_regions_interactions
                )
                for lambda_variable_name, lambda_forces in alchemical_forces.items():
                    try:
                        alchemical_forces_by_lambda[lambda_variable_name].extend(lambda_forces)
                    except KeyError:
                        alchemical_forces_by_lambda[lambda_variable_name] = lambda_forces

        # Remove original forces that have been alchemically modified.
        for force_index in reversed(forces_to_remove):
            alchemical_system.removeForce(force_index)

        # Add forces and split groups if necessary.
        self._add_alchemical_forces(alchemical_system, alchemical_forces_by_lambda)

        # Record timing statistics.
        timer.stop("Create alchemically modified system")
        timer.report_timing()

        # If the System uses a NonbondedForce, replace its NonbondedForce implementation of reaction field
        # with a Custom*Force implementation that uses c_rf = 0.
        # NOTE: This adds an additional CustomNonbondedForce
        if self.alchemical_rf_treatment == "switched":
            forcefactories.replace_reaction_field(
                alchemical_system, return_copy=False, switch_width=self.switch_width
            )

        return alchemical_system
