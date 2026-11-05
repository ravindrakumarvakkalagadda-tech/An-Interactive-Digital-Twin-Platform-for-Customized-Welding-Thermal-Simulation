"""
Welding Digital Twin — interactive prototype
Run:
    pip install -r requirements_welding_digital_twin.txt
    streamlit run welding_digital_twin_app.py
"""
import contextlib, importlib.util, io, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
SOLVER_PATH = HERE / 
if not SOLVER_PATH.exists():
    st.error("Solver file not found. Place V3.5 beside this app.")
    st.stop()

spec = importlib.util.spec_from_file_location("welding_solver_v35", SOLVER_PATH)
solver = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = solver
spec.loader.exec_module(solver)

st.set_page_config(page_title="Welding Digital Twin", page_icon="🔥", layout="wide")
st.title("Welding Digital Twin")
st.caption("Interactive 3D setup and physics-based transient thermal simulation | Prototype Milestone 1")
st.info("This prototype uses the V3.5 constant-property solver as its computational backend. The 3D preview is interactive; the first release runs the solver as a batch and then visualizes saved states.")

if "run_dir" not in st.session_state:
    st.session_state.run_dir = None

with st.sidebar:
    st.header("Simulation controls")
    st.subheader("1. Specimen geometry (mm)")
    L = st.number_input("Length", min_value=20.0, value=150.0, step=5.0)
    W = st.number_input("Width", min_value=20.0, value=80.0, step=5.0)
    H = st.number_input("Thickness", min_value=1.0, value=7.5, step=0.5)
    sx = st.number_input("Weld start from left edge", min_value=10.0, value=25.0, step=1.0)
    weld_len = st.number_input("Weld length", min_value=1.0, value=100.0, step=5.0)
    wy = st.number_input("Weld line from near edge", min_value=0.0, value=40.0, step=1.0)

    st.subheader("2. Material — constant properties")
    mat_name = st.text_input("Material name", "User-defined steel")
    T0 = st.number_input("Initial temperature (°C)", value=27.0)
    rho = st.number_input("Density (kg/m³)", min_value=1.0, value=6384.912889)
    k = st.number_input("Conductivity (W/m·K)", min_value=0.01, value=51.062)
    cp = st.number_input("Specific heat (J/kg·K)", min_value=0.01, value=571.235)
    st.caption(f"Derived diffusivity: {k/(rho*cp):.4e} m²/s")

    st.subheader("3. Welding process")
    power = st.number_input("Welding power (W)", min_value=1.0, value=3532.8)
    efficiency = st.slider("Efficiency", 0.05, 1.0, 0.85, 0.01)
    speed = st.number_input("Travel speed (m/s)", min_value=0.0001, value=0.00278, format="%.5f")
    source = st.selectbox("Heat-source model", ["goldak", "uniform_surface"])
    st.caption("Goldak semi-axes are specified below in mm.")
    af = ar = b = c = 0.0
    ff, fr = 0.4, 1.6
    if source == "goldak":
        af = st.number_input("Front semi-axis af (mm)", min_value=0.1, value=3.025)
        ar = st.number_input("Rear semi-axis ar (mm)", min_value=0.1, value=12.1)
        b = st.number_input("Transverse semi-axis b (mm)", min_value=0.1, value=4.472)
        c = st.number_input("Depth semi-axis c (mm)", min_value=0.1, value=6.05)
        ff = st.number_input("Front fraction ff", min_value=0.01, value=0.4)
        fr = st.number_input("Rear fraction fr", min_value=0.01, value=1.6)
    else:
        fp_len = st.number_input("Surface footprint length (mm)", min_value=0.1, value=10.0)
        fp_wid = st.number_input("Surface footprint width (mm)", min_value=0.1, value=8.0)

    st.subheader("4. Time and mesh")
    end_time = st.number_input("Simulation end time (s)", min_value=0.01, value=10.0)
    dt0 = st.number_input("Initial time step (s)", min_value=1e-6, value=0.0005, format="%.6f")
    dtmin = st.number_input("Minimum time step (s)", min_value=1e-8, value=0.00001, format="%.7f")
    dtmax = st.number_input("Maximum time step (s)", min_value=1e-6, value=0.005, format="%.5f")
    dx = st.number_input("Element dx (mm)", min_value=0.1, value=3.0)
    dy = st.number_input("Element dy (mm)", min_value=0.1, value=1.0)
    dz = st.number_input("Element dz (mm)", min_value=0.1, value=1.5)

    st.subheader("5. Output")
    output_times_text = st.text_input("Save times (s), comma-separated", "0,1,5,10")
    surfaces = st.multiselect("Contour surfaces", ["top","bottom","left","right","near","far"], ["top"])
    outroot = st.text_input("Output folder", "welding_digital_twin_results")
    run_clicked = st.button("Validate and run simulation", type="primary", use_container_width=True)

# Interactive geometry preview
st.subheader("Specimen and weld-path preview")
if sx + weld_len > L:
    st.error("Weld end exceeds specimen length.")
if wy > W:
    st.error("Weld line must lie within specimen width.")
fig = go.Figure()
# plate top and bottom surfaces
corners = [(0,0,0),(L,0,0),(L,W,0),(0,W,0),(0,0,H),(L,0,H),(L,W,H),(0,W,H)]
faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
fig.add_trace(go.Mesh3d(x=[p[0] for p in corners], y=[p[1] for p in corners],
    z=[p[2] for p in corners], i=[f[0] for f in faces], j=[f[1] for f in faces],
    k=[f[2] for f in faces], color="lightsteelblue", opacity=0.35, name="Specimen"))
fig.add_trace(go.Scatter3d(x=[sx,sx+weld_len], y=[wy,wy], z=[H+0.15,H+0.15],
    mode="lines", line=dict(color="red", width=8), name="Weld path"))
torch_x = min(max(sx,0),L)
fig.add_trace(go.Scatter3d(x=[torch_x], y=[wy], z=[H+1.2], mode="markers",
    marker=dict(size=7,color="orange",symbol="diamond"), name="Torch (start position)"))
fig.update_layout(height=500, scene=dict(xaxis_title="X — length (mm)", yaxis_title="Y — width (mm)",
    zaxis_title="Z — thickness (mm)", aspectmode="data"), margin=dict(l=0,r=0,t=10,b=0),
    legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Coordinate convention:** origin at the near-left-bottom corner; weld travels in +X. The torch marker shows the starting position in this prototype.")

if run_clicked:
    errors=[]
    if sx < 10: errors.append("Weld start must be at least 10 mm from the left edge.")
    if sx+weld_len > L: errors.append("Weld path extends beyond specimen length.")
    if wy > W: errors.append("Weld line lies outside specimen width.")
    if not surfaces: errors.append("Select at least one output surface.")
    try:
        save_times=[float(v.strip()) for v in output_times_text.split(",") if v.strip()]
        if not save_times or any(v<0 or v>end_time for v in save_times):
            errors.append("Save times must be within 0 and the simulation end time.")
    except Exception:
        save_times=[]
        errors.append("Enter valid comma-separated save times.")
    if not (0 < dtmin <= dt0 <= dtmax):
        errors.append("Time steps must satisfy minimum ≤ initial ≤ maximum.")
    if errors:
        for e in errors: st.error(e)
    else:
        inp=solver.SimulationInputs()
        g,m,ld,tc,el,o=inp.geometry,inp.material,inp.loading,inp.time,inp.elements,inp.output
        g.length_m=L/1000; g.width_m=W/1000; g.thickness_m=H/1000
        g.weld_start_x_m=sx/1000; g.weld_length_m=weld_len/1000; g.weld_track_y_m=wy/1000
        m.name=mat_name; m.initial_temperature_C=T0; m.density_kg_m3=rho
        m.conductivity_W_mK=k; m.specific_heat_J_kgK=cp
        ld.source_model=source; ld.welding_power_W=power; ld.efficiency=efficiency; ld.travel_speed_m_s=speed
        if source=="goldak":
            ld.goldak_front_a_m=af/1000; ld.goldak_rear_a_m=ar/1000
            ld.goldak_b_m=b/1000; ld.goldak_c_m=c/1000
            ld.goldak_front_fraction=ff; ld.goldak_rear_fraction=fr
        else:
            ld.footprint_length_m=fp_len/1000; ld.footprint_width_m=fp_wid/1000
        tc.time_step_mode="adaptive"; tc.initial_time_step_s=dt0
        tc.minimum_time_step_s=dtmin; tc.maximum_time_step_s=dtmax
        tc.simulation_end_time_s=end_time
        el.dx_m=dx/1000; el.dy_m=dy/1000; el.dz_m=dz/1000
        o.output_mode="selected_times"; o.selected_output_times_s=sorted(set(save_times))
        o.contour_surfaces=surfaces; o.output_directory=outroot
        o.save_temperature_contours=True; o.save_3d_temperature_field=True
        o.save_thermal_histories=True; o.save_energy_balance=True
        status=st.status("Running pre-checks and simulation...", expanded=True)
        try:
            # Validate before launching. The solver creates a unique run folder.
            solver.pre_simulation_check(inp, display=False)
            log=io.StringIO()
            with contextlib.redirect_stdout(log):
                solver.run_simulation(inp, prechecked=True)
            root=Path(outroot)
            runs=sorted(root.glob("run_*"), key=lambda q:q.stat().st_mtime, reverse=True)
            if not runs: raise RuntimeError("Solver finished without producing a run folder.")
            st.session_state.run_dir=str(runs[0].resolve())
            status.update(label="Simulation completed.", state="complete", expanded=False)
            with st.expander("Solver execution log"):
                st.text(log.getvalue()[-12000:])
        except Exception as exc:
            status.update(label="Simulation failed validation or execution.", state="error")
            st.error(str(exc))
            st.stop()

run_dir=st.session_state.run_dir
if run_dir and Path(run_dir).exists():
    rd=Path(run_dir)
    st.divider()
    st.subheader("Simulation results")
    summary_path=rd/"run_summary.json"
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        c1,c2,c3=st.columns(3)
        c1.metric("Peak temperature",f"{summary.get('peak_temperature_C',float('nan')):.1f} °C")
        c2.metric("Simulation steps",f"{summary.get('number_of_steps','—'):,}")
        c3.metric("Energy residual",f"{summary.get('energy_balance_residual_J',float('nan')):.2e} J")
        st.caption(f"Run folder: {rd}")
    fields=sorted(rd.glob("temperature_field_t*.npz"))
    if fields:
        selected=st.select_slider("Saved simulation time", options=fields,
            format_func=lambda p: f"{float(p.stem.split('t')[-1]):.4f} s")
        data=np.load(selected)
        T=data["T_C"]; x=data["x_m"]*1000; y=data["y_m"]*1000; z=data["z_m"]*1000
        zi=len(z)-1
        surf=go.Figure(data=go.Surface(x=x,y=y,z=T[:,:,zi].T,colorscale="Turbo",
            colorbar=dict(title="°C"), contours={"z":{"show":False}}))
        surf.update_layout(title=f"Top-surface temperature field at {float(data['time_s']):.3f} s",
            height=600, scene=dict(xaxis_title="X (mm)",yaxis_title="Y (mm)",zaxis_title="Temperature (°C)",
            aspectmode="manual",aspectratio=dict(x=1.5,y=0.8,z=0.5)))
        st.plotly_chart(surf,use_container_width=True)
    probes=rd/"probe_temperature_histories.csv"
    if probes.exists():
        df=pd.read_csv(probes)
        st.subheader("Probe temperature histories")
        chart=go.Figure()
        for pid,grp in df.groupby("probe_id"):
            chart.add_trace(go.Scatter(x=grp.time_s,y=grp.temperature_C,mode="lines",name=f"Probe {pid}"))
        chart.update_layout(xaxis_title="Time (s)",yaxis_title="Temperature (°C)",height=400)
        st.plotly_chart(chart,use_container_width=True)
    st.success("Results are saved locally in the run-specific output folder.")
