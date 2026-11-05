"""
WELDING THERMAL SIMULATION
=================================================

Complete configurable 3-D transient thermal solver.

Coordinate convention:
  Origin (0,0,0) = near-left-bottom specimen corner; X = length/weld travel,
  Y = width, Z = thickness. Interactive geometry inputs use edge distances in mm.

Workflow:
  1. Geometry
  2. Material properties
  3. Loading and boundary conditions
  4. Time increment and convergence controls
  5. Element details
  6. Output controls
  7. Pre-simulation diagnostics and solver execution

Numerical method:
  Cell-centred explicit finite-volume conduction solver.
  Constant material properties throughout each simulation. Density, thermal
  conductivity and specific heat are user inputs; alpha=k/(rho*Cp) is derived
  internally.
  Goldak double-ellipsoidal volumetric source or uniform moving surface source.
  Convection/radiation on top and side faces; contact conductance at bottom.

Important scope/limitations:
  - Single-phase sensible-heat model only. Latent heat, solid-liquid phase change,
    melting flow, keyhole physics, weld-pool convection and phase-dependent
    properties are not included. Do not interpret results above the material's
    validated single-phase temperature range as physically reliable.
  - Constant properties are an intentional simplifying assumption. The model
    does not represent temperature-dependent property changes.
  - Goldak source is discretely normalised over the plate domain. This preserves
    total power but can distort the source shape if the source is near a boundary;
    diagnostics warn about that condition.
  - Adaptive mode here enforces user limits and physical/numerical step bounds.
    It is not an embedded local-truncation-error controller.
  - Mesh recommendations are screening guidance only. Mesh independence must
    be demonstrated by refinement studies.

Dependencies:
    pip install numpy pandas matplotlib

Run:
    python welding_simulation_v2_1.py

All internal units are SI; material temperatures are in deg C.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
import json
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. USER INPUT DATA CLASSES
# =============================================================================

@dataclass
class GeometryInput:
    length_m: float = 0.150
    width_m: float = 0.080
    thickness_m: float = 0.0075
    weld_start_x_m: float = 0.025
    weld_length_m: float = 0.100
    weld_track_y_m: float = 0.040


@dataclass
class MaterialInput:
    name: str = "User-defined steel"
    initial_temperature_C: float = 27.0
    density_kg_m3: float = 6384.912889
    conductivity_W_mK: float = 51.062
    specific_heat_J_kgK: float = 571.235


@dataclass
class LoadingBoundaryInput:
    source_model: Literal["goldak", "uniform_surface"] = "goldak"
    welding_power_W: float = 3532.8
    efficiency: float = 0.85
    travel_speed_m_s: float = 0.00278

    # Goldak parameters
    goldak_b_m: float = 0.004472
    goldak_c_m: float = 0.00605
    goldak_front_a_m: float = 0.003025
    goldak_rear_a_m: float = 0.0121
    goldak_front_fraction: float = 0.4
    goldak_rear_fraction: float = 1.6

    # Uniform surface source footprint
    footprint_length_m: float = 0.010
    footprint_width_m: float = 0.008

    # User-entered boundary condition values; equations retained
    ambient_temperature_C: float = 27.0
    emissivity: float = 0.91
    convection_W_m2K: float = 6.0
    contact_conductance_W_m2K: float = 3000.0
    support_table_temperature_C: float = 27.0
    stefan_boltzmann_W_m2K4: float = 5.670374419e-8


@dataclass
class TimeConvergenceInput:
    # fixed: use initial dt subject to safety limits
    # adaptive: choose a step within min/max using safety restrictions
    time_step_mode: Literal["fixed", "adaptive"] = "adaptive"
    initial_time_step_s: float = 0.0005
    minimum_time_step_s: float = 0.00001
    maximum_time_step_s: float = 0.0050

    # Explicit stability and practical resolution controls
    explicit_safety_factor: float = 0.35
    max_source_temperature_rise_K_per_step: float = 5.0
    max_torch_travel_fraction_of_smallest_cell: float = 0.5

    # Run length: None means weld travel duration
    simulation_end_time_s: Optional[float] = None

    # Time-step convergence study (configuration only; separate runs)
    run_time_step_convergence: bool = False
    time_step_refinement_factors: list[float] = field(default_factory=lambda:
        [1.0, 0.5, 0.25])


@dataclass
class ElementInput:
    dx_m: float = 0.003
    dy_m: float = 0.001
    dz_m: float = 0.0015
    recommended_cells_per_source_scale: int = 4
    run_mesh_convergence: bool = False
    mesh_refinement_factors: list[float] = field(default_factory=lambda:
        [1.0, 0.75, 0.5])


@dataclass
class OutputInput:
    output_mode: Literal["fixed_interval", "selected_times", "both"] = "selected_times"
    output_interval_s: float = 1.0
    selected_output_times_s: list[float] = field(default_factory=lambda:
        [1, 2, 5, 10, 15, 20, 30, 36])

    save_temperature_contours: bool = True
    save_3d_temperature_field: bool = True
    save_thermal_histories: bool = True
    save_energy_balance: bool = True

    # Select complete external surfaces: top, bottom, left, right, near, far, all.
    contour_surfaces: list[str] = field(default_factory=lambda: ["top"])

    # Probe coordinates are (x, y, z) in metres. Empty list uses default probes.
    probe_coordinates_m: list[tuple[float, float, float]] = field(default_factory=list)
    output_directory: str = "welding_results_v2_6"


@dataclass
class SimulationInputs:
    geometry: GeometryInput = field(default_factory=GeometryInput)
    material: MaterialInput = field(default_factory=MaterialInput)
    loading: LoadingBoundaryInput = field(default_factory=LoadingBoundaryInput)
    time: TimeConvergenceInput = field(default_factory=TimeConvergenceInput)
    elements: ElementInput = field(default_factory=ElementInput)
    output: OutputInput = field(default_factory=OutputInput)


# =============================================================================
# 2. SEQUENTIAL INPUT INTERFACE
# =============================================================================

def ask_float(prompt: str, default: float) -> float:
    """Read one numeric value; reject comma-separated lists with clear guidance."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        if "," in raw:
            print(
                "This field accepts ONE number, not a comma-separated list. "
                "If you are entering several contour/result saving times, "
                "enter them only when prompted for 'Selected output times "
                "(seconds, comma-separated)'."
            )
            continue
        try:
            value = float(raw)
            if not np.isfinite(value):
                print("Please enter a finite numeric value.")
                continue
            return value
        except ValueError:
            print("Invalid numeric value. Please enter one number, for example 0.0025.")


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return default if raw == "" else int(raw)


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    raw = input(f"{prompt} {choices} [{default}]: ").strip().lower()
    value = default if raw == "" else raw
    if value not in choices:
        raise ValueError(f"Choose one of: {', '.join(choices)}")
    return value


def ask_bool(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} ({suffix}): ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "1", "true")


def enter_geometry(g: GeometryInput) -> GeometryInput:
    print("\n" + "=" * 72 + "\nMODULE 1 - GEOMETRY\n" + "=" * 72)
    print("""
COORDINATE SYSTEM AND WELD-PATH CONVENTION
------------------------------------------
The specimen origin is defined automatically at the NEAR-LEFT-BOTTOM corner.

                         Y (width)
                         ^
          FAR edge       +-------------------------------+
                         |                               |
                         |          WELD PATH            |
                         |          ----------------> X  |
                         |                               |
          NEAR edge      +-------------------------------+----> X (length)
                       (0,0,0)

Z is through the plate thickness, positive upward from the bottom face.
The weld travels in the +X direction.

You do NOT need to enter X/Y/Z coordinates. Enter distances from the
specimen edges in millimetres.
""")
    g.length_m = ask_float("Specimen length (mm)", g.length_m * 1000.0) / 1000.0
    g.width_m = ask_float("Specimen width (mm)", g.width_m * 1000.0) / 1000.0
    g.thickness_m = ask_float("Specimen thickness (mm)", g.thickness_m * 1000.0) / 1000.0

    g.weld_start_x_m = ask_float(
        "Weld start distance from LEFT edge (mm; minimum 10 mm)",
        g.weld_start_x_m * 1000.0
    ) / 1000.0
    g.weld_length_m = ask_float(
        "Weld bead length along the X direction (mm)",
        g.weld_length_m * 1000.0
    ) / 1000.0
    g.weld_track_y_m = ask_float(
        "Weld-line distance measured from NEAR side edge (mm)",
        g.weld_track_y_m * 1000.0
    ) / 1000.0

    print("\nGeometry interpretation:")
    print("  Origin: near-left-bottom corner (0, 0, 0)")
    print(f"  Weld starts {g.weld_start_x_m*1000:.3f} mm from the left end edge")
    print(f"  Weld line is {g.weld_track_y_m*1000:.3f} mm from the near side edge")
    print(f"  Weld direction: +X; bead length = {g.weld_length_m*1000:.3f} mm")
    print(f"  Weld ends {((g.weld_start_x_m+g.weld_length_m)*1000):.3f} mm from the left edge")
    return g


def enter_material(m: MaterialInput) -> MaterialInput:
    print("\n" + "=" * 72 + "\nMODULE 2 - CONSTANT MATERIAL PROPERTIES (V3.5)\n" + "=" * 72)
    m.name = input(f"Material name [{m.name}]: ").strip() or m.name
    m.initial_temperature_C = ask_float("Initial material temperature (deg C)", m.initial_temperature_C)
    print("""
CONSTANT-PROPERTY MODEL
The same density, thermal conductivity and specific heat are used at every
simulation temperature. Thermal diffusivity is calculated internally:
alpha = k/(rho*Cp). No temperature-property table is required.
""")
    m.density_kg_m3 = ask_float("Density (kg/m3)", m.density_kg_m3)
    m.conductivity_W_mK = ask_float("Thermal conductivity (W/m K)", m.conductivity_W_mK)
    m.specific_heat_J_kgK = ask_float("Specific heat capacity (J/kg K)", m.specific_heat_J_kgK)
    if min(m.density_kg_m3, m.conductivity_W_mK, m.specific_heat_J_kgK) <= 0:
        raise ValueError("Density, conductivity and specific heat must all be positive.")
    print("Constant material properties accepted.")
    return m


def enter_loading(g: LoadingBoundaryInput) -> LoadingBoundaryInput:
    print("\n" + "=" * 72 + "\nMODULE 3 - LOADING AND BOUNDARY CONDITIONS\n" + "=" * 72)
    g.source_model = ask_choice("Heat-source model", ["goldak", "uniform_surface"], g.source_model)
    g.welding_power_W = ask_float("Welding power (W)", g.welding_power_W)
    g.efficiency = ask_float("Heat-source efficiency (0-1)", g.efficiency)
    g.travel_speed_m_s = ask_float("Welding speed (m/s)", g.travel_speed_m_s)

    if g.source_model == "goldak":
        print(r"""
GOLDAK DOUBLE-ELLIPSOIDAL HEAT-SOURCE PARAMETERS
------------------------------------------------
The parameters below define the size and power distribution of the moving
volumetric heat source. All geometric dimensions are SEMI-AXES (radii),
not full lengths.

                         +X: torch travel direction
             REAR (-X)          |          FRONT (+X)
          <---------------------|---------------------->
                    ar          |          af
                 (rear)         |        (front)

                         Weld centreline
                              |
                         <--- b --->
                       transverse width

                 c = penetration/depth semi-axis,
                 measured downward from the TOP weld surface.

Coordinate convention in this software:
  X: specimen length / torch travel
  Y: specimen width
  Z=thickness: top weld surface
  Z=0: bottom surface

Power split:
  ff = front ellipsoid power fraction parameter
  fr = rear ellipsoid power fraction parameter
  Under the convention implemented here, ff + fr is expected to equal 2.
  This is a model convention check; use values consistent with your
  chosen Goldak formulation and reference, rather than treating it as
  a universal rule for every published parameterisation.

Enter dimensions in MILLIMETRES. The program converts them internally to metres.
""")
        print("Current Goldak parameter defaults:")
        print(f"  af (front longitudinal semi-axis) = {g.goldak_front_a_m*1000:.3f} mm")
        print(f"  ar (rear longitudinal semi-axis)  = {g.goldak_rear_a_m*1000:.3f} mm")
        print(f"  b  (transverse semi-axis)         = {g.goldak_b_m*1000:.3f} mm")
        print(f"  c  (depth semi-axis)              = {g.goldak_c_m*1000:.3f} mm")
        print(f"  ff (front fraction parameter)     = {g.goldak_front_fraction:.3f}")
        print(f"  fr (rear fraction parameter)      = {g.goldak_rear_fraction:.3f}")
        print(f"  ff + fr                           = {g.goldak_front_fraction + g.goldak_rear_fraction:.3f} (expected 2 for this implementation)")

        g.goldak_front_a_m = ask_float(
            "af - front longitudinal semi-axis (mm)", g.goldak_front_a_m*1000.0
        ) / 1000.0
        g.goldak_rear_a_m = ask_float(
            "ar - rear longitudinal semi-axis (mm)", g.goldak_rear_a_m*1000.0
        ) / 1000.0
        g.goldak_b_m = ask_float(
            "b - transverse semi-axis (mm)", g.goldak_b_m*1000.0
        ) / 1000.0
        g.goldak_c_m = ask_float(
            "c - depth/penetration semi-axis (mm)", g.goldak_c_m*1000.0
        ) / 1000.0
        g.goldak_front_fraction = ask_float(
            "ff - front ellipsoid fraction parameter", g.goldak_front_fraction
        )
        g.goldak_rear_fraction = ask_float(
            "fr - rear ellipsoid fraction parameter", g.goldak_rear_fraction
        )

        fraction_sum = g.goldak_front_fraction + g.goldak_rear_fraction
        print("\nGoldak parameter review:")
        print(f"  af = {g.goldak_front_a_m*1000:.3f} mm | ar = {g.goldak_rear_a_m*1000:.3f} mm")
        print(f"  b  = {g.goldak_b_m*1000:.3f} mm | c  = {g.goldak_c_m*1000:.3f} mm")
        print(f"  ff = {g.goldak_front_fraction:.3f} | fr = {g.goldak_rear_fraction:.3f}")
        print(f"  ff + fr = {fraction_sum:.3f}")
        if not np.isclose(fraction_sum, 2.0, atol=1e-6):
            print("  CAUTION: ff + fr is not 2 for the convention used by this implementation.")
            print("  Verify the parameters against the Goldak formulation/reference you intend to reproduce.")
    else:
        g.footprint_length_m = ask_float("Uniform footprint length (m)", g.footprint_length_m)
        g.footprint_width_m = ask_float("Uniform footprint width (m)", g.footprint_width_m)

    print("\nBoundary-condition inputs (existing formulations are retained):")
    g.ambient_temperature_C = ask_float("Ambient temperature (deg C)", g.ambient_temperature_C)
    g.emissivity = ask_float("Surface emissivity", g.emissivity)
    g.convection_W_m2K = ask_float("Convection coefficient (W/m2 K)", g.convection_W_m2K)
    g.contact_conductance_W_m2K = ask_float("Bottom contact conductance (W/m2 K)", g.contact_conductance_W_m2K)
    g.support_table_temperature_C = ask_float("Support-table temperature (deg C)", g.support_table_temperature_C)
    return g


def enter_time_controls(t: TimeConvergenceInput) -> TimeConvergenceInput:
    print("\n" + "=" * 72 + "\nMODULE 4 - TIME INCREMENT AND CONVERGENCE\n" + "=" * 72)
    t.time_step_mode = ask_choice("Time-step mode", ["fixed", "adaptive"], t.time_step_mode)
    t.initial_time_step_s = ask_float("Initial time increment (s)", t.initial_time_step_s)
    t.minimum_time_step_s = ask_float("Minimum time increment (s)", t.minimum_time_step_s)
    t.maximum_time_step_s = ask_float("Maximum time increment (s)", t.maximum_time_step_s)
    t.explicit_safety_factor = ask_float("Explicit stability safety factor (0-1)", t.explicit_safety_factor)
    t.max_source_temperature_rise_K_per_step = ask_float(
        "Maximum estimated source temperature rise per step (K)",
        t.max_source_temperature_rise_K_per_step)
    t.max_torch_travel_fraction_of_smallest_cell = ask_float(
        "Maximum torch travel per step (fraction of smallest cell)",
        t.max_torch_travel_fraction_of_smallest_cell)
    raw = input("Simulation end time in seconds [Enter = weld travel duration]: ").strip()
    t.simulation_end_time_s = None if raw == "" else float(raw)
    return t


def enter_elements(e: ElementInput) -> ElementInput:
    print("\n" + "=" * 72 + "\nMODULE 5 - ELEMENT DETAILS\n" + "=" * 72)
    print("Enter element dimensions in millimetres (mm). Values are converted to metres internally.")
    e.dx_m = ask_float("Element size dx (mm)", e.dx_m * 1000.0) / 1000.0
    e.dy_m = ask_float("Element size dy (mm)", e.dy_m * 1000.0) / 1000.0
    e.dz_m = ask_float("Element size dz (mm)", e.dz_m * 1000.0) / 1000.0
    e.recommended_cells_per_source_scale = ask_int(
        "Guideline cells across source characteristic length",
        e.recommended_cells_per_source_scale)
    return e


def enter_output_controls(o: OutputInput) -> OutputInput:
    print("\n" + "=" * 72 + "\nOUTPUT CONTROL\n" + "=" * 72)
    o.output_mode = ask_choice("Output timing mode",
                               ["fixed_interval", "selected_times", "both"], o.output_mode)
    if o.output_mode in ("fixed_interval", "both"):
        o.output_interval_s = ask_float(
            "OUTPUT INTERVAL: enter ONE interval in seconds (e.g. 1.0)",
            o.output_interval_s
        )
    print("For multiple individual save times, enter a comma-separated list at the next prompt.")
    raw = input("SELECTED OUTPUT TIMES: enter times in seconds separated by commas "
                f"(e.g. 0,1,5,10,20) [{','.join(map(str, o.selected_output_times_s))}]: ").strip()
    if raw:
        o.selected_output_times_s = [float(v.strip()) for v in raw.split(",") if v.strip()]

    o.save_temperature_contours = ask_bool("Save temperature contours", o.save_temperature_contours)
    o.save_3d_temperature_field = ask_bool("Save 3-D temperature fields", o.save_3d_temperature_field)
    o.save_thermal_histories = ask_bool("Save probe thermal histories", o.save_thermal_histories)
    o.save_energy_balance = ask_bool("Save energy balance history", o.save_energy_balance)

    allowed = ["top", "bottom", "left", "right", "near", "far", "all"]
    print("Surface options: top, bottom, left, right, near, far, all")
    print("The selected COMPLETE surface will be saved at each requested output time.")
    raw = input(f"Surfaces to save (comma-separated) [{','.join(o.contour_surfaces)}]: ").strip()
    if raw:
        o.contour_surfaces = [v.strip().lower() for v in raw.split(",") if v.strip()]
        invalid = set(o.contour_surfaces) - set(allowed)
        if invalid:
            raise ValueError(f"Invalid surface selection(s): {sorted(invalid)}")
    o.output_directory = input(f"Output directory [{o.output_directory}]: ").strip() or o.output_directory
    return o


# =============================================================================
# 3. MATERIAL PROPERTIES AND DIAGNOSTICS
# =============================================================================

def material_diagnostics(m: MaterialInput) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate and report the single constant-property material definition."""
    values = [m.density_kg_m3, m.conductivity_W_mK, m.specific_heat_J_kgK]
    errors, messages = [], []
    labels = ["Density", "Thermal conductivity", "Specific heat capacity"]
    for label, value in zip(labels, values):
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            errors.append(f"{label} must be finite and positive.")
    if errors:
        return pd.DataFrame(), errors, messages
    rho, k, cp = map(float, values)
    alpha = k/(rho*cp)
    df = pd.DataFrame([{
        "property_model": "constant",
        "density_kg_m3": rho,
        "conductivity_W_mK": k,
        "specific_heat_J_kgK": cp,
        "thermal_diffusivity_m2_s": alpha
    }])
    messages.append("INFO: Constant density, conductivity and specific heat are used throughout the simulation.")
    messages.append("INFO: Thermal diffusivity is derived internally as k/(rho*Cp).")
    return df, errors, messages


class MaterialModel:
    """Constant-property sensible-heat material model."""
    def __init__(self, m: MaterialInput):
        self.input = m
        self.df, errors, self.messages = material_diagnostics(m)
        if errors:
            raise ValueError("Material input errors: " + " | ".join(errors))
        self.rho_const = float(m.density_kg_m3)
        self.k_const = float(m.conductivity_W_mK)
        self.cp_const = float(m.specific_heat_J_kgK)
        self.alpha_const = self.k_const/(self.rho_const*self.cp_const)
        self.temp = np.array([float(m.initial_temperature_C)])
        self.rho = np.array([self.rho_const])
        self.k = np.array([self.k_const])
        self.cp = np.array([self.cp_const])
        self.alpha = np.array([self.alpha_const])

    def properties(self, T):
        arr = np.asarray(T, dtype=float)
        shape = arr.shape
        return (np.full(shape, self.k_const), np.full(shape, self.cp_const),
                np.full(shape, self.rho_const), np.full(shape, self.alpha_const))

    def enthalpy(self, T):
        return self.rho_const*self.cp_const*np.asarray(T, dtype=float)

    def temperature_from_enthalpy(self, H):
        return np.asarray(H, dtype=float)/(self.rho_const*self.cp_const)


def validate_inputs(inp: SimulationInputs, mat: MaterialModel) -> list[str]:
    """Return all blocking errors and non-blocking warnings before solver launch."""
    g, x, t, e, o = inp.geometry, inp.loading, inp.time, inp.elements, inp.output
    errors, warnings_list, infos = [], [], []

    def finite_positive(name, value, allow_zero=False):
        try:
            v = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric.")
            return
        if not math.isfinite(v) or (v < 0 if allow_zero else v <= 0):
            qualifier = "finite and non-negative" if allow_zero else "finite and positive"
            errors.append(f"{name} must be {qualifier}.")

    # Geometry and weld path.
    for name, value in [("Specimen length", g.length_m),
                        ("Specimen width", g.width_m),
                        ("Specimen thickness", g.thickness_m)]:
        finite_positive(name, value)
    for name, value in [("Weld start distance", g.weld_start_x_m),
                        ("Weld length", g.weld_length_m),
                        ("Weld-line distance from near edge", g.weld_track_y_m)]:
        finite_positive(name, value, allow_zero=(name == "Weld start distance"))
    if g.weld_start_x_m < 0.010 - 1e-12:
        errors.append("Weld source must start at least 10 mm from the left edge.")
    if g.length_m > 0 and g.weld_length_m > 0:
        if g.weld_start_x_m + g.weld_length_m > g.length_m + 1e-12:
            errors.append("Weld path extends beyond the right specimen edge.")
    if g.width_m > 0 and not (0 <= g.weld_track_y_m <= g.width_m):
        errors.append("Weld track centreline must lie within the specimen width.")

    # Constant material-property validation.
    for label, value in [("density", inp.material.density_kg_m3),
                         ("conductivity", inp.material.conductivity_W_mK),
                         ("specific heat", inp.material.specific_heat_J_kgK)]:
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            errors.append(f"Constant {label} must be finite and positive.")
    if not math.isfinite(inp.material.initial_temperature_C):
        errors.append("Initial material temperature must be finite.")

    # Loading and boundary conditions.
    finite_positive("Welding power", x.welding_power_W)
    finite_positive("Welding travel speed", x.travel_speed_m_s)
    if not math.isfinite(x.efficiency) or not (0 < x.efficiency <= 1):
        errors.append("Heat-source efficiency must be greater than 0 and no greater than 1.")
    if x.source_model not in ("goldak", "uniform_surface"):
        errors.append("Heat-source model must be 'goldak' or 'uniform_surface'.")
    if not math.isfinite(x.ambient_temperature_C) or not math.isfinite(x.support_table_temperature_C):
        errors.append("Ambient and support-table temperatures must be finite.")
    if not math.isfinite(x.emissivity) or not (0 <= x.emissivity <= 1):
        errors.append("Surface emissivity must be between 0 and 1.")
    finite_positive("Convection coefficient", x.convection_W_m2K, allow_zero=True)
    finite_positive("Bottom contact conductance", x.contact_conductance_W_m2K, allow_zero=True)
    finite_positive("Stefan-Boltzmann constant", x.stefan_boltzmann_W_m2K4)
    if x.source_model == "goldak":
        for name, value in [
            ("Goldak front semi-axis", x.goldak_front_a_m),
            ("Goldak rear semi-axis", x.goldak_rear_a_m),
            ("Goldak width semi-axis", x.goldak_b_m),
            ("Goldak depth semi-axis", x.goldak_c_m),
            ("Goldak front fraction", x.goldak_front_fraction),
            ("Goldak rear fraction", x.goldak_rear_fraction)
        ]:
            finite_positive(name, value)
        if not np.isclose(x.goldak_front_fraction + x.goldak_rear_fraction,
                          2.0, atol=1e-6):
            warnings_list.append(
                "Goldak front and rear fractions do not sum to 2; verify the convention."
            )
        if all(v > 0 for v in [x.goldak_front_a_m, x.goldak_rear_a_m,
                               x.goldak_b_m, x.goldak_c_m]):
            if g.weld_start_x_m < 3*x.goldak_rear_a_m:
                warnings_list.append(
                    "Goldak source may be clipped near the start edge; consider a longer approach path."
                )
            right_margin = g.length_m - (g.weld_start_x_m + g.weld_length_m)
            if right_margin < 3*x.goldak_front_a_m:
                warnings_list.append(
                    "Goldak source may be clipped near the weld end; consider a run-off margin."
                )
            side_margin = min(g.weld_track_y_m, g.width_m-g.weld_track_y_m)
            if side_margin < 3*x.goldak_b_m:
                warnings_list.append(
                    "Goldak source is close to a side edge; source truncation may occur."
                )
    elif x.source_model == "uniform_surface":
        finite_positive("Uniform-source footprint length", x.footprint_length_m)
        finite_positive("Uniform-source footprint width", x.footprint_width_m)
        if g.weld_start_x_m - x.footprint_length_m/2 < 0:
            errors.append("Uniform heat-source footprint is clipped at the left edge at t=0.")
        if (g.weld_start_x_m + g.weld_length_m + x.footprint_length_m/2) > g.length_m:
            errors.append("Uniform heat-source footprint is clipped at the right edge at weld end.")
        if (g.weld_track_y_m - x.footprint_width_m/2 < 0 or
                g.weld_track_y_m + x.footprint_width_m/2 > g.width_m):
            errors.append("Uniform heat-source footprint extends beyond a side edge.")

    # Time integration and convergence controls.
    if t.time_step_mode not in ("fixed", "adaptive"):
        errors.append("Time-step mode must be 'fixed' or 'adaptive'.")
    for name, value in [
        ("Initial time increment", t.initial_time_step_s),
        ("Minimum time increment", t.minimum_time_step_s),
        ("Maximum time increment", t.maximum_time_step_s),
        ("Maximum source temperature rise per step", t.max_source_temperature_rise_K_per_step),
        ("Maximum torch travel fraction", t.max_torch_travel_fraction_of_smallest_cell)
    ]:
        finite_positive(name, value)
    if not (0 < t.minimum_time_step_s <= t.initial_time_step_s <= t.maximum_time_step_s):
        errors.append("Time increments must satisfy 0 < minimum <= initial <= maximum.")
    if not math.isfinite(t.explicit_safety_factor) or not (0 < t.explicit_safety_factor <= 1):
        errors.append("Explicit stability safety factor must be in (0, 1].")
    if t.simulation_end_time_s is not None:
        finite_positive("Simulation end time", t.simulation_end_time_s)
    if t.run_time_step_convergence:
        factors = np.asarray(t.time_step_refinement_factors, dtype=float)
        if len(factors) < 2 or not np.isfinite(factors).all() or np.any(factors <= 0):
            errors.append("Time-step refinement factors must contain at least two positive finite values.")

    # Mesh and element details.
    for name, value in [("Element size dx", e.dx_m),
                        ("Element size dy", e.dy_m),
                        ("Element size dz", e.dz_m)]:
        finite_positive(name, value)
    if not isinstance(e.recommended_cells_per_source_scale, (int, np.integer)) or e.recommended_cells_per_source_scale < 1:
        errors.append("Recommended cells across source scale must be a positive integer.")
    if e.run_mesh_convergence:
        factors = np.asarray(e.mesh_refinement_factors, dtype=float)
        if len(factors) < 2 or not np.isfinite(factors).all() or np.any(factors <= 0):
            errors.append("Mesh refinement factors must contain at least two positive finite values.")

    # Output requests and destination.
    if o.output_mode not in ("fixed_interval", "selected_times", "both"):
        errors.append("Output timing mode is invalid.")
    if o.output_mode in ("fixed_interval", "both"):
        finite_positive("Output interval", o.output_interval_s)
    allowed_surfaces = {"top", "bottom", "left", "right", "near", "far", "all"}
    if not o.contour_surfaces or set(o.contour_surfaces) - allowed_surfaces:
        errors.append("Select one or more valid output surfaces: top, bottom, left, right, near, far, all.")
    if not isinstance(o.output_directory, str) or not o.output_directory.strip():
        errors.append("Output directory must be a non-empty path.")
    if o.output_mode in ("selected_times", "both"):
        try:
            requested = np.asarray(o.selected_output_times_s, dtype=float)
            if not np.isfinite(requested).all():
                errors.append("Selected output times must all be finite numbers.")
            elif np.any(requested < 0):
                errors.append("Selected output times cannot be negative.")
            else:
                end_time = (t.simulation_end_time_s if t.simulation_end_time_s is not None
                            else g.weld_length_m/x.travel_speed_m_s)
                if np.any(requested > end_time + 1e-10):
                    errors.append(
                        f"Selected output times must not exceed simulation end time ({end_time:g} s)."
                    )
                if len(np.unique(requested)) != len(requested):
                    warnings_list.append("Duplicate selected output times were supplied; duplicates will be saved once.")
        except (TypeError, ValueError):
            errors.append("Selected output times must be a list of numeric values.")
    for i, probe in enumerate(o.probe_coordinates_m):
        if len(probe) != 3 or not np.isfinite(np.asarray(probe, dtype=float)).all():
            errors.append(f"Probe {i+1} must contain three finite coordinates (x, y, z).")
        elif not (0 <= probe[0] <= g.length_m and
                  0 <= probe[1] <= g.width_m and
                  0 <= probe[2] <= g.thickness_m):
            errors.append(f"Probe {i+1} lies outside the specimen geometry.")

    # Estimate the discretisation size and guard against invalid/extreme grids.
    if all(v > 0 and math.isfinite(v) for v in
           [g.length_m, g.width_m, g.thickness_m, e.dx_m, e.dy_m, e.dz_m]):
        counts = [max(1, int(round(g.length_m/e.dx_m))),
                  max(1, int(round(g.width_m/e.dy_m))),
                  max(1, int(round(g.thickness_m/e.dz_m)))]
        n_cells = math.prod(counts)
        if n_cells > 50_000_000:
            errors.append(
                f"Requested mesh contains about {n_cells:,} cells, exceeding the "
                "50-million-cell safety limit. Increase element sizes."
            )
        elif n_cells > 5_000_000:
            warnings_list.append(
                f"Requested mesh contains about {n_cells:,} cells and may require substantial memory."
            )

    if errors:
        return ["ERROR: " + item for item in errors]
    if warnings_list:
        infos.extend("WARNING: " + item for item in warnings_list)
    infos.append(
        f"INFO: Net heat input = {x.efficiency*x.welding_power_W:.3f} W "
        "(efficiency applied once)."
    )
    return infos


class PreSimulationValidationError(ValueError):
    """Aggregated pre-simulation validation failure."""


def pre_simulation_check(inp: SimulationInputs, display: bool = True):
    """Validate all inputs and perform non-solver prechecks before execution."""
    if display:
        print("\n" + "="*72)
        print("PRE-SIMULATION INPUT VALIDATION")
        print("="*72)
        print("Checking geometry, material data, loading, boundary conditions,")
        print("time controls, mesh details and requested output settings...")

    try:
        mat = MaterialModel(inp.material)
    except Exception as exc:
        raise PreSimulationValidationError(
            f"Material-property input is invalid: {exc}"
        ) from None

    messages = validate_inputs(inp, mat)
    blocking = [m for m in messages if m.startswith("ERROR:")]
    if blocking:
        raise PreSimulationValidationError("; ".join(m.replace("ERROR: ", "") for m in blocking))

    # Warnings are visible but do not block execution.
    for message in messages:
        if display:
            print(message)

    try:
        grid = make_grid(inp.geometry, inp.elements)
        details = mesh_and_time_diagnostics(inp, mat, grid)
        end_time = (inp.time.simulation_end_time_s
                    if inp.time.simulation_end_time_s is not None
                    else inp.geometry.weld_length_m/inp.loading.travel_speed_m_s)
        # Validate source integration at the beginning and end of the requested
        # weld path, before creating run folders or entering the transient solver.
        weld_travel_duration = (
            inp.geometry.weld_length_m / inp.loading.travel_speed_m_s
        )
        source_end_check_time = min(end_time, weld_travel_duration)
        for test_time in (0.0, source_end_check_time):
            build_source(
                grid[3], grid[4], grid[5], test_time, inp,
                grid[6], grid[7], grid[8]
            )
        # The initial numerical step must be capable of satisfying the user's
        # minimum step and the computed explicit stability / torch-motion limits.
        q0, _, _, _ = build_source(
            grid[3], grid[4], grid[5], 0.0, inp,
            grid[6], grid[7], grid[8]
        )
        dt_selected, step_details = choose_time_step(
            inp, mat, grid, q0, inp.time.initial_time_step_s
        )
        details.update(step_details)
        details["prechecked_initial_time_step_s"] = float(dt_selected)
        details["estimated_cell_count"] = int(np.prod(details["cell_counts"]))
    except Exception as exc:
        raise PreSimulationValidationError(
            f"Pre-simulation numerical check failed: {exc}"
        ) from None

    if display:
        print("\nPRE-SIMULATION NUMERICAL CHECK:")
        print(json.dumps(details, indent=2))
        print("\n" + "="*72)
        print("ALL INPUT PARAMETERS PASSED VALIDATION.")
        print("No blocking input errors were detected.")
        print("The simulation is ready for execution.")
        print("="*72)
    return mat, grid, details


def make_grid(g: GeometryInput, e: ElementInput):
    nx = max(1, int(round(g.length_m / e.dx_m)))
    ny = max(1, int(round(g.width_m / e.dy_m)))
    nz = max(1, int(round(g.thickness_m / e.dz_m)))
    dx, dy, dz = g.length_m / nx, g.width_m / ny, g.thickness_m / nz
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    z = (np.arange(nz) + 0.5) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z, dx, dy, dz


def mesh_and_time_diagnostics(inp: SimulationInputs, mat: MaterialModel, grid):
    g, x, t, e = inp.geometry, inp.loading, inp.time, inp.elements
    xx, yy, zz, X, Y, Z, dx, dy, dz = grid
    alpha_max = float(np.max(mat.alpha))
    dt_diff = t.explicit_safety_factor / (
        2 * alpha_max * (1/dx**2 + 1/dy**2 + 1/dz**2)
    )

    if x.source_model == "goldak":
        scales = [
            ("front longitudinal full scale", 2*x.goldak_front_a_m, dx),
            ("rear longitudinal full scale", 2*x.goldak_rear_a_m, dx),
            ("transverse full scale", 2*x.goldak_b_m, dy),
            ("depth full scale", 2*x.goldak_c_m, dz),
        ]
        source_lengths = [2*x.goldak_front_a_m, 2*x.goldak_rear_a_m,
                          2*x.goldak_b_m, 2*x.goldak_c_m]
    else:
        scales = [
            ("footprint length", x.footprint_length_m, dx),
            ("footprint width", x.footprint_width_m, dy)
        ]
        source_lengths = [x.footprint_length_m, x.footprint_width_m]

    cells = [length / h for _, length, h in scales]
    smallest_cell = min(dx, dy, dz)
    dt_motion = (t.max_torch_travel_fraction_of_smallest_cell * smallest_cell /
                 x.travel_speed_m_s)

    return {
        "cell_counts": [len(xx), len(yy), len(zz)],
        "effective_element_sizes_m": [dx, dy, dz],
        "alpha_max_m2_s": alpha_max,
        "diffusion_stability_limit_s": dt_diff,
        "torch_motion_limit_s": dt_motion,
        "source_scale_names": [v[0] for v in scales],
        "source_scale_cells": cells,
        "smallest_source_scale_cells": min(cells),
        "smallest_cell_m": smallest_cell,
        "diffusion_length_at_initial_dt_m": math.sqrt(alpha_max*t.initial_time_step_s),
        "torch_travel_at_initial_dt_m": x.travel_speed_m_s*t.initial_time_step_s,
        "gross_welding_power_W": x.welding_power_W,
        "net_heat_input_W": x.efficiency*x.welding_power_W
    }


# =============================================================================
# 5. HEAT SOURCE MODELS
# =============================================================================

def goldak_source(X, Y, Z, time_s, g: GeometryInput, x: LoadingBoundaryInput):
    """Goldak double-ellipsoidal source applied from the TOP surface.

    Coordinate convention: Z=0 is the bottom face and Z=t is the top face.
    Therefore the depth measured downward from the weld surface is d=t-Z.
    +X is the forward direction of torch travel; -X is the rear region.
    """
    power = x.efficiency * x.welding_power_W
    xs = g.weld_start_x_m + x.travel_speed_m_s * time_s
    xi = X - xs
    eta_y = Y - g.weld_track_y_m
    depth_from_top = g.thickness_m - Z
    q = np.zeros_like(X)
    front = xi >= 0.0
    rear = ~front

    pref_f = (6*np.sqrt(3)*x.goldak_front_fraction*power /
              (np.pi**1.5*x.goldak_front_a_m*x.goldak_b_m*x.goldak_c_m))
    pref_r = (6*np.sqrt(3)*x.goldak_rear_fraction*power /
              (np.pi**1.5*x.goldak_rear_a_m*x.goldak_b_m*x.goldak_c_m))

    q[front] = pref_f*np.exp(-3*(
        xi[front]**2/x.goldak_front_a_m**2 +
        eta_y[front]**2/x.goldak_b_m**2 +
        depth_from_top[front]**2/x.goldak_c_m**2))
    q[rear] = pref_r*np.exp(-3*(
        xi[rear]**2/x.goldak_rear_a_m**2 +
        eta_y[rear]**2/x.goldak_b_m**2 +
        depth_from_top[rear]**2/x.goldak_c_m**2))
    return q, xs, power


def uniform_surface_source(X, Y, Z, time_s, g: GeometryInput,
                           x: LoadingBoundaryInput, dx, dy, dz):
    power = x.efficiency*x.welding_power_W
    xs = g.weld_start_x_m + x.travel_speed_m_s*time_s
    x_centres = X[:, 0, 0]
    y_centres = Y[0, :, 0]
    xl, xr = x_centres-dx/2, x_centres+dx/2
    yl, yr = y_centres-dy/2, y_centres+dy/2
    sx0, sx1 = xs-x.footprint_length_m/2, xs+x.footprint_length_m/2
    sy0, sy1 = g.weld_track_y_m-x.footprint_width_m/2, g.weld_track_y_m+x.footprint_width_m/2
    ox = np.maximum(0, np.minimum(xr, sx1)-np.maximum(xl, sx0))
    oy = np.maximum(0, np.minimum(yr, sy1)-np.maximum(yl, sy0))
    area = ox[:, None]*oy[None, :]
    nominal = x.footprint_length_m*x.footprint_width_m
    covered = float(area.sum())
    if not np.isclose(covered, nominal, rtol=1e-8, atol=1e-12):
        raise ValueError("Uniform heat-source footprint is clipped by specimen boundary. "
                         "Adjust weld path or footprint.")
    q = np.zeros_like(X)
    q[:, :, 0] = (power/nominal)*area/(dx*dy*dz)
    return q, xs, power


def build_source(X, Y, Z, time_s, inp, dx, dy, dz):
    g, x = inp.geometry, inp.loading
    if x.source_model == "uniform_surface":
        q, xs, power = uniform_surface_source(X, Y, Z, time_s, g, x, dx, dy, dz)
        return q, xs, power, 1.0

    q, xs, power = goldak_source(X, Y, Z, time_s, g, x)
    cell_volume = dx*dy*dz
    integrated_raw = float(np.sum(q)*cell_volume)
    if integrated_raw <= 0:
        raise ValueError("Goldak source has zero integrated power on the computational grid.")

    # For this Goldak convention, the theoretical integrated power is
    # ((ff + fr)/2) * Q. Compare the discretely captured power against that
    # value to detect clipping by plate boundaries / insufficient domain.
    fraction_sum = (x.goldak_front_fraction + x.goldak_rear_fraction) / 2.0
    theoretical_power = power * fraction_sum
    domain_capture_fraction = integrated_raw / theoretical_power if theoretical_power > 0 else 0.0

    # Retain energy-normalisation for the solver's energy accounting, but expose
    # the required correction factor and warn when the source is strongly clipped.
    q *= power / integrated_raw
    return q, xs, power, domain_capture_fraction


# =============================================================================
# 6. FINITE-VOLUME OPERATORS AND BOUNDARY CONDITIONS
# =============================================================================

def conductive_divergence(T, k, dx, dy, dz):
    div = np.zeros_like(T)
    kf = 2*k[1:,:,:]*k[:-1,:,:]/(k[1:,:,:]+k[:-1,:,:]+1e-30)
    gx = kf*(T[1:,:,:]-T[:-1,:,:])/dx
    div[:-1,:,:] += gx/dx
    div[1:,:,:] -= gx/dx

    kf = 2*k[:,1:,:]*k[:,:-1,:]/(k[:,1:,:]+k[:,:-1,:]+1e-30)
    gy = kf*(T[:,1:,:]-T[:,:-1,:])/dy
    div[:,:-1,:] += gy/dy
    div[:,1:,:] -= gy/dy

    kf = 2*k[:,:,1:]*k[:,:,:-1]/(k[:,:,1:]+k[:,:,:-1]+1e-30)
    gz = kf*(T[:,:,1:]-T[:,:,:-1])/dz
    div[:,:,:-1] += gz/dz
    div[:,:,1:] -= gz/dz
    return div


def boundary_loss(T, k, inp, dx, dy, dz):
    x = inp.loading
    TinfK = x.ambient_temperature_C + 273.15
    TK = np.maximum(T+273.15, 1.0)
    loss = np.zeros_like(T)

    # Top exposed face is z = thickness (last z-index).
    qtop = (x.convection_W_m2K*(T[:,:,-1]-x.ambient_temperature_C) +
            x.emissivity*x.stefan_boltzmann_W_m2K4*(TK[:,:,-1]**4-TinfK**4))
    loss[:,:,-1] += qtop/dz

    # Bottom face is z = 0 (first z-index), in contact with support table.
    kbot = np.maximum(k[:,:,0], 1e-12)
    h_eff = 1/(1/x.contact_conductance_W_m2K + dz/(2*kbot))
    qcontact = h_eff*(T[:,:,0]-x.support_table_temperature_C)
    loss[:,:,0] += qcontact/dz

    qx0 = (x.convection_W_m2K*(T[0,:,:]-x.ambient_temperature_C) +
           x.emissivity*x.stefan_boltzmann_W_m2K4*(TK[0,:,:]**4-TinfK**4))
    qx1 = (x.convection_W_m2K*(T[-1,:,:]-x.ambient_temperature_C) +
           x.emissivity*x.stefan_boltzmann_W_m2K4*(TK[-1,:,:]**4-TinfK**4))
    loss[0,:,:] += qx0/dx
    loss[-1,:,:] += qx1/dx

    qy0 = (x.convection_W_m2K*(T[:,0,:]-x.ambient_temperature_C) +
           x.emissivity*x.stefan_boltzmann_W_m2K4*(TK[:,0,:]**4-TinfK**4))
    qy1 = (x.convection_W_m2K*(T[:,-1,:]-x.ambient_temperature_C) +
           x.emissivity*x.stefan_boltzmann_W_m2K4*(TK[:,-1,:]**4-TinfK**4))
    loss[:,0,:] += qy0/dy
    loss[:,-1,:] += qy1/dy
    return loss


# =============================================================================
# 7. TIME-STEP CONTROL
# =============================================================================

def source_peak_heating_limit(qvol, mat: MaterialModel, T):
    # Conservative local volumetric heat-capacity lower bound over table.
    cv_min = float(np.min(mat.rho*mat.cp))
    qpeak = float(np.max(qvol))
    if qpeak <= 0:
        return float("inf")
    # Bound source-only temperature increase per step.
    return 1.0  # replaced by caller using configured rise threshold


def calculate_time_step_limits(inp, mat, grid, qvol):
    diag = mesh_and_time_diagnostics(inp, mat, grid)
    t = inp.time
    cv_min = float(np.min(mat.rho*mat.cp))
    qpeak = float(np.max(qvol))
    dt_source = (t.max_source_temperature_rise_K_per_step*cv_min/qpeak
                 if qpeak > 0 else float("inf"))
    dt_motion = diag["torch_motion_limit_s"]
    dt_stable = min(diag["diffusion_stability_limit_s"], dt_source, dt_motion,
                    t.maximum_time_step_s)
    return dt_stable, {
        **diag,
        "source_heating_limit_s": dt_source,
        "combined_numerical_limit_s": dt_stable
    }


def choose_time_step(inp, mat, grid, qvol, current_dt):
    t = inp.time
    dt_stable, details = calculate_time_step_limits(inp, mat, grid, qvol)
    if dt_stable < t.minimum_time_step_s:
        raise RuntimeError(
            f"Required stable time step {dt_stable:.6g} s is below user minimum "
            f"{t.minimum_time_step_s:.6g} s. Refine mesh only after reviewing "
            "the stability condition, increase minimum-step allowance, or use "
            "a different numerical method."
        )
    if t.time_step_mode == "fixed":
        requested = t.initial_time_step_s
        if requested > dt_stable:
            raise RuntimeError(
                f"Fixed initial time step {requested:.6g} s exceeds combined "
                f"limit {dt_stable:.6g} s. Reduce initial dt or select adaptive mode."
            )
        return requested, details
    # Adaptive: current dt may grow up to initial/max, but is always clamped
    # to the current physical/numerical limits.
    target_cap = min(t.maximum_time_step_s, dt_stable)
    dt = min(max(current_dt, t.minimum_time_step_s), target_cap)
    return dt, details


# =============================================================================
# 8. INTERPOLATION, PROBES AND OUTPUT
# =============================================================================

def bracket(grid, value):
    if value < grid[0] or value > grid[-1]:
        raise ValueError(f"Coordinate {value} outside cell-centre range [{grid[0]}, {grid[-1]}].")
    hi = int(np.searchsorted(grid, value, side="left"))
    if hi == 0:
        return 0, 0, 0.0
    if hi >= len(grid):
        return len(grid)-1, len(grid)-1, 0.0
    lo = hi-1
    return lo, hi, float((value-grid[lo])/(grid[hi]-grid[lo]))


def sample_trilinear(T, x, y, z, xq, yq, zq):
    i0,i1,fx = bracket(x,xq); j0,j1,fy = bracket(y,yq); k0,k1,fz = bracket(z,zq)
    c00 = T[i0,j0,k0]*(1-fx)+T[i1,j0,k0]*fx
    c10 = T[i0,j1,k0]*(1-fx)+T[i1,j1,k0]*fx
    c01 = T[i0,j0,k1]*(1-fx)+T[i1,j0,k1]*fx
    c11 = T[i0,j1,k1]*(1-fx)+T[i1,j1,k1]*fx
    c0 = c00*(1-fy)+c10*fy
    c1 = c01*(1-fy)+c11*fy
    return float(c0*(1-fz)+c1*fz)


def default_probe_coordinates(inp, x_grid, y_grid, z_grid):
    """Place default thermal probes along the weld path near the top surface.

    Five fixed probes are distributed along the weld length. Each coordinate
    is snapped to the nearest cell centre so it is valid for interpolation.
    """
    g = inp.geometry
    fractions = [0.05, 0.25, 0.50, 0.75, 0.95]
    z_target = float(z_grid[-1])
    y_target = float(y_grid[int(np.argmin(np.abs(y_grid-g.weld_track_y_m)))])
    coords = []
    for frac in fractions:
        x_target = g.weld_start_x_m + frac*g.weld_length_m
        x_target = float(x_grid[int(np.argmin(np.abs(x_grid-x_target)))])
        coords.append((x_target, y_target, z_target))
    return coords


def output_times(inp, end_time):
    o = inp.output
    values = {0.0, float(end_time)}
    if o.output_mode in ("fixed_interval", "both"):
        n = int(math.floor(end_time/o.output_interval_s))
        values.update(i*o.output_interval_s for i in range(1,n+1))
    if o.output_mode in ("selected_times", "both"):
        values.update(v for v in o.selected_output_times_s if 0 <= v <= end_time)
    return sorted(values)


def selected_surface_fields(T, inp):
    """Extract selected complete external surfaces for later common-scale plotting."""
    selected = set(inp.output.contour_surfaces)
    if "all" in selected:
        selected = {"top", "bottom", "left", "right", "near", "far"}

    surfaces = {
        "top": (T[:, :, -1].T.copy(), "Top surface", "X (length)", "Y (width)"),
        "bottom": (T[:, :, 0].T.copy(), "Bottom surface", "X (length)", "Y (width)"),
        "left": (T[0, :, :].T.copy(), "Left end surface", "Y (width)", "Z (thickness)"),
        "right": (T[-1, :, :].T.copy(), "Right end surface", "Y (width)", "Z (thickness)"),
        "near": (T[:, 0, :].T.copy(), "Near side surface", "X (length)", "Z (thickness)"),
        "far": (T[:, -1, :].T.copy(), "Far side surface", "X (length)", "Z (thickness)")
    }
    return {name: surfaces[name] for name in selected if name in surfaces}


def save_surface_contours(surface_data, x, y, z, time_s, outdir, vmin, vmax):
    """Render selected surfaces with identical colour limits for every saved time."""
    axes = {
        "top": (x, y), "bottom": (x, y),
        "left": (y, z), "right": (y, z),
        "near": (x, z), "far": (x, z)
    }
    tag = f"{time_s:09.4f}".replace(".", "p")
    for name, (field, title, xlabel, ylabel) in surface_data.items():
        coord1, coord2 = axes[name]
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            field, origin="lower", aspect="auto",
            extent=[coord1[0]*1000, coord1[-1]*1000,
                    coord2[0]*1000, coord2[-1]*1000],
            cmap="inferno", vmin=vmin, vmax=vmax
        )
        ax.set_xlabel(f"{xlabel} (mm)")
        ax.set_ylabel(f"{ylabel} (mm)")
        ax.set_title(f"{title} temperature contour, t = {time_s:.4f} s")
        fig.colorbar(im, ax=ax, label="Temperature (deg C)")
        fig.tight_layout()
        fig.savefig(outdir / f"surface_{name}_t{tag}.png", dpi=180)
        plt.close(fig)


# =============================================================================
# 9. SOLVER
# =============================================================================

def run_simulation(inp: SimulationInputs, prechecked=False):
    # Never create a run folder or start solver work until all inputs pass.
    if prechecked:
        mat, grid, precheck_details = pre_simulation_check(inp, display=False)
    else:
        mat, grid, precheck_details = pre_simulation_check(inp, display=True)

    g, m_in, load, tc, elem, out_cfg = (
        inp.geometry, inp.material, inp.loading, inp.time, inp.elements, inp.output
    )
    output_root = Path(out_cfg.output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outdir = output_root / f"run_{run_stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    # Store the actual run-specific path in the configuration and summary.
    out_cfg.output_directory = str(outdir)
    save_input_configuration(inp, outdir/"input_configuration.json")

    x, y, z, X, Y, Z, dx, dy, dz = grid
    T = np.full(X.shape, m_in.initial_temperature_C, dtype=float)
    peak_temperature_ever_C = float(np.max(T))
    peak_temperature_ever_location_m = [
        float(x[0]), float(y[0]), float(z[0])
    ]
    peak_temperature_ever_time_s = 0.0

    weld_duration = g.weld_length_m/load.travel_speed_m_s
    end_time = tc.simulation_end_time_s if tc.simulation_end_time_s is not None else weld_duration
    if end_time <= 0:
        raise ValueError("Simulation end time must be positive.")

    # Requested output times are landed on exactly by shortening the preceding step.
    save_times = output_times(inp, end_time)
    save_set = set(round(v, 12) for v in save_times)

    probes = out_cfg.probe_coordinates_m or default_probe_coordinates(inp, x, y, z)
    for px,py,pz in probes:
        if not (0 <= px <= g.length_m and 0 <= py <= g.width_m and 0 <= pz <= g.thickness_m):
            raise ValueError(f"Probe coordinate {(px,py,pz)} lies outside specimen geometry.")

    print("\n=== MATERIAL PROPERTY TABLE (derived diffusivity) ===")
    print(mat.df.to_string(index=False))
    for msg in mat.messages:
        print(msg)

    q0, _, _, capture0 = build_source(X,Y,Z,0.0,inp,dx,dy,dz)
    dt_first, step_details = choose_time_step(inp,mat,grid,q0,tc.initial_time_step_s)
    print("\n=== MESH / TIME-STEP PRE-CHECK ===")
    for key, value in step_details.items():
        print(f"{key}: {value}")
    print(f"selected_initial_dt_s: {dt_first:.8g}")
    print(f"initial_goldak_domain_capture_fraction: {capture0:.4f}")
    print(f"rough_step_count_estimate_at_initial_dt: {math.ceil(end_time/dt_first):,}")
    if capture0 < 0.90:
        print("WARNING: More than 10% of the theoretical Goldak source is clipped by the specimen domain at t=0; discrete renormalisation is being applied. Increase start/side/end margins or model an approach/run-off path.")

    min_source_cells = step_details["smallest_source_scale_cells"]
    if min_source_cells < elem.recommended_cells_per_source_scale:
        print(
            f"WARNING: Only {min_source_cells:.2f} cells span the smallest "
            "source characteristic scale. Refine mesh and perform mesh convergence."
        )

    # Enthalpy-based storage update for temperature-dependent volumetric heat capacity.
    H = mat.enthalpy(T)
    t_now = 0.0
    current_dt = min(tc.initial_time_step_s, tc.maximum_time_step_s)
    step = 0
    output_index = 0
    probe_rows, energy_rows, step_rows = [], [], []
    contour_snapshots = []
    cumulative_input = 0.0
    cumulative_loss = 0.0
    cumulative_storage = 0.0
    cell_volume = dx*dy*dz

    def record_state(time_value):
        nonlocal output_index
        Tmax = float(np.max(T)); Tmean = float(np.mean(T))
        if out_cfg.save_thermal_histories:
            for pidx,(px,py,pz) in enumerate(probes):
                # Coordinates at physical faces are clamped to nearest cell-centre range.
                pxs = float(np.clip(px,x[0],x[-1]))
                pys = float(np.clip(py,y[0],y[-1]))
                pzs = float(np.clip(pz,z[0],z[-1]))
                temp = sample_trilinear(T,x,y,z,pxs,pys,pzs)
                probe_rows.append([time_value,pidx,px,py,pz,temp])
        residual = cumulative_storage-(cumulative_input-cumulative_loss)
        energy_rows.append([time_value,Tmax,Tmean,cumulative_input,cumulative_loss,
                            cumulative_storage,residual])
        if out_cfg.save_temperature_contours:
            contour_snapshots.append(
                (float(time_value), selected_surface_fields(T, inp))
            )
        if out_cfg.save_3d_temperature_field:
            np.savez_compressed(outdir/f"temperature_field_t{time_value:09.4f}.npz",
                                T_C=T,x_m=x,y_m=y,z_m=z,time_s=time_value)
        output_index += 1

    record_state(0.0)

    print(
        "\\nNUMERICAL STABILITY MONITOR: progress reports show the maximum and "
        "RMS temperature-field update per accepted time step, together with "
        "the global energy-balance residual. This is a transient explicit "
        "time-marching scheme, not an iterative solver; these indicators "
        "monitor numerical behaviour and do not constitute iterative "
        "convergence proof."
    )

    while t_now < end_time - 1e-12:
        qvol, source_x, net_power, source_capture = build_source(X,Y,Z,t_now,inp,dx,dy,dz)
        dt, details = choose_time_step(inp,mat,grid,qvol,current_dt)

        # Land exactly on the next requested output time or simulation end time.
        future_targets = [v for v in save_times if v > t_now + 1e-11]
        next_target = min(future_targets) if future_targets else end_time
        dt = min(dt, next_target-t_now, end_time-t_now)

        if dt < tc.minimum_time_step_s - 1e-14:
            # A shorter terminal/output-alignment step is permitted; ordinary steps are not.
            if (next_target-t_now) < tc.minimum_time_step_s:
                pass
            else:
                raise RuntimeError("Required time increment fell below user minimum.")
        if dt <= 0:
            break

        k, cp, rho, alpha = mat.properties(T)
        div_cond = conductive_divergence(T,k,dx,dy,dz)
        qloss = boundary_loss(T,k,inp,dx,dy,dz)

        # Enthalpy update: dH/dt = div(k grad T) + q_source - q_boundary.
        net_rate = div_cond + qvol - qloss
        H_new = H + dt*net_rate
        T_new = mat.temperature_from_enthalpy(H_new)

        if not np.isfinite(T_new).all():
            raise FloatingPointError("Non-finite temperature encountered.")

        # Explicit transient update monitor (not an iterative solver residual).
        max_field_update_K = float(np.max(np.abs(T_new - T)))
        rms_field_update_K = float(np.sqrt(np.mean((T_new - T)**2)))

        applied_power = float(np.sum(qvol)*cell_volume)
        boundary_loss_power = float(np.sum(qloss)*cell_volume)
        storage_increment = float(np.sum(H_new-H)*cell_volume)
        cumulative_input += applied_power*dt
        cumulative_loss += boundary_loss_power*dt
        cumulative_storage += storage_increment

        # Global energy-accounting residual, updated at every accepted step.
        energy_residual_J = cumulative_storage - (cumulative_input - cumulative_loss)

        T = T_new
        H = H_new
        t_now += dt
        step += 1

        step_peak = float(np.max(T))
        if step_peak > peak_temperature_ever_C:
            peak_idx = np.unravel_index(int(np.argmax(T)), T.shape)
            peak_temperature_ever_C = step_peak
            peak_temperature_ever_location_m = [
                float(x[peak_idx[0]]),
                float(y[peak_idx[1]]),
                float(z[peak_idx[2]])
            ]
            peak_temperature_ever_time_s = float(t_now)
        step_rows.append([step,t_now,dt,float(T.max()),source_x,
                          applied_power,boundary_loss_power,source_capture,
                          max_field_update_K,rms_field_update_K,energy_residual_J])

        # Simple bounded growth for adaptive mode. It does not estimate local truncation error.
        if tc.time_step_mode == "adaptive":
            current_dt = min(tc.maximum_time_step_s, max(tc.minimum_time_step_s, dt*1.25))
        else:
            current_dt = tc.initial_time_step_s

        if abs(t_now-next_target) <= 1e-9 or abs(t_now-end_time) <= 1e-9:
            t_now = next_target if abs(t_now-next_target) <= 1e-9 else t_now
            record_state(t_now)

        if step % 500 == 0:
            progress = min(100.0, 100.0*t_now/end_time)
            print(
                f"Progress {progress:6.2f}% | step {step:>8d} | "
                f"t={t_now:.4f}/{end_time:.4f} s | dt={dt:.3e} s | "
                f"max|Delta T|={max_field_update_K:.3e} K | "
                f"RMS(Delta T)={rms_field_update_K:.3e} K | "
                f"energy residual={energy_residual_J:.3e} J | "
                f"source capture={source_capture:.3f}",
                flush=True
            )

    # Render all saved contours using one common temperature scale.
    if out_cfg.save_temperature_contours and contour_snapshots:
        all_values = [
            field
            for _, surface_data in contour_snapshots
            for field, _, _, _ in surface_data.values()
        ]
        common_vmin = min(float(np.min(a)) for a in all_values)
        common_vmax = max(float(np.max(a)) for a in all_values)
        if common_vmax <= common_vmin:
            common_vmax = common_vmin + 1.0
        print(
            f"\nContour colour scale shared across all saved times/surfaces: "
            f"{common_vmin:.2f} to {common_vmax:.2f} deg C"
        )
        for saved_time, surface_data in contour_snapshots:
            save_surface_contours(
                surface_data, x, y, z, saved_time, outdir,
                common_vmin, common_vmax
            )
        with open(outdir/"contour_colour_scale.json", "w", encoding="utf-8") as f:
            json.dump({
                "vmin_C": common_vmin,
                "vmax_C": common_vmax,
                "basis": "global minimum and maximum across all selected surfaces at saved output times"
            }, f, indent=2)

    # Save tabular results
    if out_cfg.save_thermal_histories:
        pd.DataFrame(probe_rows,columns=[
            "time_s","probe_id","x_m","y_m","z_m","temperature_C"
        ]).to_csv(outdir/"probe_temperature_histories.csv",index=False)

    if out_cfg.save_energy_balance:
        pd.DataFrame(energy_rows,columns=[
            "time_s","maximum_temperature_C","mean_temperature_C",
            "cumulative_heat_input_J","cumulative_boundary_loss_J",
            "cumulative_enthalpy_change_J","energy_balance_residual_J"
        ]).to_csv(outdir/"energy_balance.csv",index=False)

    pd.DataFrame(step_rows,columns=[
        "step","time_s","dt_s","maximum_temperature_C","source_x_m",
        "applied_power_W","boundary_loss_W","source_domain_capture_fraction",
        "max_temperature_field_update_K","rms_temperature_field_update_K",
        "energy_balance_residual_J"
    ]).to_csv(outdir/"time_step_history.csv",index=False)

    mat.df.to_csv(outdir/"material_properties_with_derived_diffusivity.csv",index=False)

    max_index = np.unravel_index(int(np.argmax(T)), T.shape)
    max_location_m = [float(x[max_index[0]]), float(y[max_index[1]]), float(z[max_index[2]])]
    summary = {
        "software_version": "3.5",
        "material_name": m_in.name,
        "results_directory": str(outdir.resolve()),
        "contour_colour_scale": (
            {"vmin_C": common_vmin, "vmax_C": common_vmax}
            if out_cfg.save_temperature_contours and contour_snapshots else None
        ),
        "grid_cells": list(T.shape),
        "effective_element_sizes_m": [dx,dy,dz],
        "time_step_mode": tc.time_step_mode,
        "initial_time_step_s": tc.initial_time_step_s,
        "minimum_time_step_s": tc.minimum_time_step_s,
        "maximum_time_step_s": tc.maximum_time_step_s,
        "simulation_end_time_s": end_time,
        "number_of_steps": step,
        "number_of_saved_states": output_index,
        "peak_temperature_C": peak_temperature_ever_C,
        "peak_temperature_location_m": peak_temperature_ever_location_m,
        "peak_temperature_location_mm": [
            v*1000.0 for v in peak_temperature_ever_location_m
        ],
        "peak_temperature_time_s": peak_temperature_ever_time_s,
        "final_temperature_field_max_C": float(T.max()),
        "final_temperature_field_max_location_m": max_location_m,
        "gross_welding_power_W": load.welding_power_W,
        "net_heat_input_W": load.efficiency*load.welding_power_W,
        "heat_source_model": load.source_model,
        "selected_output_surfaces": out_cfg.contour_surfaces,
        "default_probe_strategy": "five probes distributed along weld length at top-surface cell centres",
        "material_property_model": "constant",
        "energy_balance_residual_J": float(cumulative_storage-(cumulative_input-cumulative_loss)),
        "energy_accounting": "Sensible enthalpy using constant volumetric heat capacity rho*Cp",
        "material_model": "constant",
        "thermal_diffusivity": "Constant; derived internally as k/(rho*Cp)",
        "latent_heat_included": False,
        "phase_change_model": "Not included; single-phase sensible heat only",
        "material_property_interpolation": "Not applicable; constant properties are used",
        "temperature_property_policy": "No temperature-dependent property table; constant values used at all temperatures",
        "boundary_conditions": "Original convection/radiation and bottom contact formulations retained; user inputs",
        "limitations": [
            "No latent heat or solid-liquid phase-change treatment is included.",
            "No weld-pool fluid flow, keyhole or free-surface physics.",
            "Goldak source is applied from the top surface using depth = plate_thickness - Z; discrete normalisation is applied and source-domain capture is reported."
        ]
    }
    with open(outdir/"run_summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)

    print("\n=== SIMULATION COMPLETE ===")
    print(json.dumps(summary,indent=2))
    print(f"Results directory: {outdir.resolve()}")
    return T, summary


# =============================================================================
# 10. INPUT CONFIGURATION SAVE/LOAD
# =============================================================================

def save_input_configuration(inp: SimulationInputs, path: Path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(asdict(inp),f,indent=2)


def run_interactive():
    inp = SimulationInputs()
    print("\nWELDING THERMAL SIMULATION FRAMEWORK v3.5")
    print("Press Enter to accept each displayed default value.")
    enter_geometry(inp.geometry)
    enter_material(inp.material)
    enter_loading(inp.loading)
    enter_time_controls(inp.time)
    enter_elements(inp.elements)
    enter_output_controls(inp.output)

    print("\n=== PRE-SIMULATION REVIEW ===")
    mat, grid, details = pre_simulation_check(inp, display=True)

    proceed = ask_bool("All checks passed. Proceed with simulation?", True)
    if not proceed:
        cfg_path = Path(inp.output.output_directory)
        cfg_path.mkdir(parents=True, exist_ok=True)
        save_input_configuration(inp, cfg_path/"input_configuration_cancelled.json")
        print(f"SIMULATION CANCELLED. Configuration saved in: {cfg_path.resolve()}")
        return

    print("\nINPUT VALIDATION SUCCESSFUL — STARTING SIMULATION.")
    run_simulation(inp, prechecked=True)


if __name__ == "__main__":
    try:
        run_interactive()
    except KeyboardInterrupt:
        print("\nSIMULATION INTERRUPTED BY USER.")
    except Exception as exc:
        # One concise user-facing error; deliberately suppress traceback.
        message = str(exc).strip().replace("\n", " ")
        print(f"\nSIMULATION NOT COMPLETED — ERROR: {message}")
