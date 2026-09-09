# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import numpy as np
import pytest
from numpy.testing import assert_equal
from openff.packmol import UNIT_CUBE
from openff.units import unit

from pontibus.utils.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)


class TestInterchangeFFSettings:
    @pytest.fixture(scope="class")
    def settings(self):
        return InterchangeFFSettings()

    def test_defaults(self, settings):
        assert_equal(settings.forcefields, ["openff-2.0.0.offxml", "tip3p.offxml"])
        assert settings.nonbonded_method == "pme"
        assert settings.nonbonded_cutoff == 0.9 * unit.nanometer
        assert settings.switch_width == 0.1 * unit.nanometer

    @pytest.mark.parametrize(
        "attr, value",
        [
            ["nonbonded_cutoff", -1.0 * unit.nanometer],
            ["switch_width", -0.1 * unit.nanometer],
            ["hydrogen_mass", -3],
        ],
    )
    def test_positives(self, settings, attr, value):
        with pytest.raises(ValueError, match="must be a positive"):
            setattr(settings, attr, value)


class TestPackmolSolvationSettings:
    @pytest.fixture()
    def settings(self):
        return PackmolSolvationSettings()

    def test_defaults(self, settings):
        assert settings.number_of_solvent_molecules is None
        assert settings.box_vectors is None
        assert settings.solvent_padding == 1.2 * unit.nanometer
        assert settings.box_shape == "cube"
        assert not settings.assign_solvent_charges
        assert settings.packing_tolerance == 2.0 * unit.angstrom
        assert settings.target_density == 0.95 * unit.grams / unit.mL

    def test_negative_solvent(self):
        with pytest.raises(ValueError, match="must be positive"):
            _ = PackmolSolvationSettings(number_of_solvent_molecules=-2, solvent_padding=None)

    def test_num_mols_and_padding(self, settings):
        msg = "Only one of ``number_solvent_molecules`` or ``solvent_padding`` can be defined"
        with pytest.raises(ValueError, match=msg):
            settings.number_of_solvent_molecules = 2

    def test_box_vectors_and_padding(self):
        msg = "Only one of ``box_vectors`` or ``solvent_padding`` can be defined."
        with pytest.raises(ValueError, match=msg):
            _ = PackmolSolvationSettings(
                box_vectors=UNIT_CUBE * unit.nanometer,
                solvent_padding=1.2 * unit.nanometer,
                target_density=None,
                box_shape=None,
            )

    def test_box_vectors_and_density(self):
        msg = "Only one of ``target_density`` or ``box_vectors`` can be defined"
        with pytest.raises(ValueError, match=msg):
            _ = PackmolSolvationSettings(
                box_vectors=UNIT_CUBE * unit.nanometer,
                solvent_padding=None,
                target_density=0.95 * unit.grams / unit.mL,
                number_of_solvent_molecules=2,
            )

    def test_density_no_shape(self, settings):
        msg = "``target_density`` and ``box_shape`` must both be defined"
        with pytest.raises(ValueError, match=msg):
            settings.box_shape = None

    def test_bad_vectors(self):
        bad_vector = (
            np.asarray(
                [
                    [0.5, 0.5, np.sqrt(2.0) / 2.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
            )
            * unit.nanometer
        )
        with pytest.raises(ValueError, match="not in OpenMM reduced form"):
            _ = PackmolSolvationSettings(
                box_vectors=bad_vector,
                solvent_padding=None,
                target_density=None,
                box_shape=None,
            )
