# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
Shared pontibus Settings.
"""

from typing import Annotated, Literal, TypeAlias

from gufe.settings import BaseForceFieldSettings
from gufe.settings.typing import (
    BoxQuantity,
    GufeQuantity,
    NanometerQuantity,
    specify_quantity_units,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BaseSolvationSettings,
)
from openff.packmol._packmol import _box_vectors_are_in_reduced_form
from openff.units import unit
from pydantic import field_validator, model_validator

GramsPerMolQuantity: TypeAlias = Annotated[GufeQuantity, specify_quantity_units("grams / mL")]
"""
Pydantic type that requires a ``pint.Quantity`` compatible with grams / mL.
Input will be converter to grams / mL upon model validation.
"""


class InterchangeFFSettings(BaseForceFieldSettings):
    """
    Parameters to set up the force field using Interchange and the
    OpenFF toolkit.
    """

    hydrogen_mass: float = 3.023841
    """Mass to be repartitioned to hydrogens from neighbouring
       heavy atoms (in amu), default 3.0"""

    # TODO; work out if we could swap these out with str of ffxml contents
    # if necessary
    forcefields: list[str] = [
        "openff-2.0.0.offxml",
        "tip3p.offxml",
    ]
    """List of force field ffxmls to apply"""

    nonbonded_method: Literal["pme", "nocutoff"] = "pme"
    """
    Method for treating nonbonded interactions, currently only PME and
    NoCutoff are allowed. Default PME.
    """

    nonbonded_cutoff: NanometerQuantity = 0.9 * unit.nanometer  # noqa: F821
    """
    Cutoff value for short range nonbonded interactions.
    Default 1.0 * unit.nanometer.
    """

    switch_width: NanometerQuantity = 0.1 * unit.nanometer  # noqa: F821
    """
    The width over which the VdW switching function is applied.
    Default 0.1 * unit.nanometer.
    """

    @field_validator("nonbonded_method")
    def allowed_nonbonded(cls, v):
        # TODO: switch to literal?
        if v.lower() not in ["pme", "nocutoff"]:
            errmsg = "Only PME and NoCutoff are allowed nonbonded methods"
            raise ValueError(errmsg)
        return v

    @field_validator("hydrogen_mass", "nonbonded_cutoff", "switch_width")
    def is_positive(cls, v):
        if v <= 0:
            errmsg = "must be a positive value"
            raise ValueError(errmsg)
        return v


class BasePontibusSolvationSettings(BaseSolvationSettings):
    """
    Base settings for use with Pontibus
    """

    number_of_solvent_molecules: int | None = None
    """
    The number of solvent molecules to add.

    Note
    ----
    * Cannot be defined alongside ``solvent_padding``.
    """

    box_vectors: BoxQuantity | None = None  # noqa: F821
    """
    Simulation box vectors.

    Note
    ----
    * Cannot be defined alongside ``target_density``.
    * If defined, ``number_of_solvent_molecules`` must be defined.
    """

    solvent_padding: NanometerQuantity | None = 1.2 * unit.nanometer  # noqa: F821
    """
    Minimum distance from any solute bounding sphere to the edge of the box.

    Note
    ----
    * Cannot be defined if ``number_of_solvent_molecules`` is defined.
    """

    box_shape: Literal["cube", "dodecahedron"] | None = "cube"
    """
    The shape of the periodic box to create.
    """

    assign_solvent_charges: bool = False
    """
    If ``True``, assign solvent charges based on the input solvent
    molecule. If ``False``, rely on library charges.

    Notes
    -----
    * If no partial charges are set in the input molecule, the molecule
      will be charged using the approach defined in ``partial_charge_settings``.
    * If not using ``ExtendedSolventComponent``, the input molecule will
      be created using ``SolventComponent.smiles`` and partial charges will
      be set using the approach defined in ``partial_charge_settings``.
    """

    packing_tolerance: NanometerQuantity = 0.2 * unit.nanometer  # noqa: F821
    """
    Packmol setting; minimum spacing between molecules in units of distance.
    2.0 A is recommended when packing proteins, but can go as low as 0.5 A
    to help with convergence.
    """

    target_density: GramsPerMolQuantity | None = 0.95 * unit.grams / unit.mL  # noqa: F821
    """
    Target mass density for the solvated system in units compatible with g / mL.
    A ``target_density`` value of 0.95 * unit.grams / unit.mL works in many cases,
    but may lead to an overly dense system. This is especially true when
    defining ``number_of_solvent_molecules``, in this case a value of
    0.715 * unit.grams / unit.mL is recommended.

    Default: 0.95 * unit.grams / unit.mL.

    Notes
    -----
    * Cannot be defined alongside ``box_vectors``
    """

    @field_validator("number_of_solvent_molecules")
    def positive_solvent_number(cls, v):
        if v is None:
            return v

        if v <= 0:
            errmsg = f"number_of_solvent molecules: {v} must be positive"
            raise ValueError(errmsg)

        return v

    @field_validator("box_vectors")
    def supported_vectors(cls, v):
        if v is not None:
            if not _box_vectors_are_in_reduced_form(v):
                errmsg = f"box_vectors: {v} are not in OpenMM reduced form"
                raise ValueError(errmsg)
        return v

    @model_validator(mode="after")
    def check_num_mols_or_padding(self):
        if not (self.number_of_solvent_molecules is None) ^ (self.solvent_padding is None):
            msg = "Only one of ``number_solvent_molecules`` or ``solvent_padding`` can be defined"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def check_target_density_or_box_vectors(self):
        if not (self.target_density is None) ^ (self.box_vectors is None):
            msg = "Only one of ``target_density`` or ``box_vectors`` can be defined"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def check_target_density_and_box_shape(self):
        if not (self.target_density is None) == (self.box_shape is None):
            msg = "``target_density`` and ``box_shape`` must both be defined"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def check_solvent_padding_or_box_vectors(self):
        if (self.box_vectors is not None) and (self.solvent_padding is not None):
            msg = "Only one of ``box_vectors`` or ``solvent_padding`` can be defined."
            raise ValueError(msg)

        return self


class PackmolSolvationSettings(BasePontibusSolvationSettings):
    """
    Settings defining how to solvate the system using Packmol.

    Notes
    -----
    * This is currently limited to the options allowed by
      Interchange's ``solvate_topology_nonwater``.
    """


class InterchangeOpenMMSolvationSettings(BasePontibusSolvationSettings):
    """
    Settings defining how to solvate the system using OpenMM.

    Notes
    -----
    * This settings class uses Interchange for system
    """
