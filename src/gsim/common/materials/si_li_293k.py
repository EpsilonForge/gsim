"""Li 293 K crystalline silicon model."""

from pdk_schema import Coord, Index, TableData, TabulatedValue

from gsim.common.materials._helpers import material_card, wavelength_validity

_LI_293K_DATA = (
    (1.20, 3.5167),
    (1.22, 3.5133),
    (1.24, 3.5102),
    (1.26, 3.5072),
    (1.28, 3.5043),
    (1.30, 3.5016),
    (1.32, 3.4990),
    (1.34, 3.4965),
    (1.36, 3.4941),
    (1.38, 3.4918),
    (1.40, 3.4896),
    (1.45, 3.4845),
    (1.50, 3.4799),
    (1.55, 3.4757),
    (1.60, 3.4719),
    (1.65, 3.4684),
    (1.70, 3.4653),
    (1.80, 3.4597),
    (1.90, 3.4550),
    (2.00, 3.4510),
    (2.25, 3.4431),
    (2.50, 3.4375),
    (2.75, 3.4334),
    (3.00, 3.4302),
    (4.00, 3.4229),
    (5.00, 3.4195),
    (6.00, 3.4177),
    (7.00, 3.4165),
    (8.00, 3.4158),
    (9.00, 3.4153),
    (10.0, 3.4150),
    (11.0, 3.4147),
    (12.0, 3.4145),
    (13.0, 3.4144),
    (14.0, 3.4142),
)

SI_LI_293K = material_card(
    name="Si-Li-293K",
    temperature_ref=293.0,
    permittivity=Index(
        validity=wavelength_validity(1.2, 14.0),
        variation=None,
        conductivity=None,
        n=TabulatedValue(
            unit="",
            data=TableData(
                dims=("wavelength",),
                coords={
                    "wavelength": Coord(
                        values=[row[0] for row in _LI_293K_DATA],
                        unit="um",
                    )
                },
                values=[row[1] for row in _LI_293K_DATA],
                attrs={},
                interp="linear",
            ),
        ),
        k=None,
    ),
)

__all__ = ["SI_LI_293K"]
