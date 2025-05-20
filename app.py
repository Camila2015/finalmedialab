import os
import io
import zipfile
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import SimpleITK as sitk
from skimage.transform import resize
import plotly.graph_objects as go

# streamlit-plotly-events must be installed:
# pip install streamlit-plotly-events
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide", page_title="Brachyanalysis - Click 3D Points")

st.markdown("""
<style>
    .giant-title { color: #28aec5; text-align: center; font-size: 72px; margin: 30px 0; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .sub-header { color: #c0d711; font-size: 24px; margin-bottom: 15px; font-weight: bold; }
    .stButton>button { background-color: #28aec5; color: white; border: none; border-radius: 4px; padding: 8px 16px; }
    .stButton>button:hover { background-color: #1c94aa; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

def find_dicom_series(directory):
    series_found = []
    for root, dirs, files in os.walk(directory):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            for sid in series_ids:
                file_list = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, sid)
                if file_list:
                    series_found.append((sid, root, file_list))
        except Exception:
            continue
    return series_found

def apply_window_level(image, window_width, window_center):
    img_float = image.astype(float)
    min_v = window_center - window_width / 2.0
    max_v = window_center + window_width / 2.0
    windowed = np.clip(img_float, min_v, max_v)
    if max_v != min_v:
        return (windowed - min_v) / (max_v - min_v)
    return np.zeros_like(img_float)

dirname = None
if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    dirname = temp_dir
    st.sidebar.success("Archivos extraídos correctamente.")

dicom_series = None
img = None
original_image = None
if dirname:
    with st.spinner('Buscando series DICOM...'):
        dicom_series = find_dicom_series(dirname)
    if dicom_series:
        options = [f"Serie {i + 1}: {series[0][:10]}... ({len(series[2])} archivos)" for i, series in enumerate(dicom_series)]
        selection = st.sidebar.selectbox("Seleccionar serie DICOM:", options)
        selected_idx = options.index(selection)
        sid, dirpath, files = dicom_series[selected_idx]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        data = reader.Execute()
        img = sitk.GetArrayViewFromImage(data)
        original_image = img
    else:
        st.sidebar.error("No se encontraron DICOM válidos en el ZIP cargado.")

if img is not None:
    n_ax, n_cor, n_sag = img.shape
    min_val, max_val = float(img.min()), float(img.max())
    default_ww = max_val - min_val
    default_wc = min_val + default_ww / 2
    ww, wc = default_ww, default_wc

    corte = st.sidebar.radio("Selecciona el tipo de corte", ("Axial", "Coronal", "Sagital"))

    if corte == "Axial":
        corte_idx = st.sidebar.slider("Selecciona el índice axial", 0, n_ax - 1, n_ax // 2)
        axial_img = img[corte_idx, :, :]
        coronal_img = img[:, n_cor // 2, :]
        sagital_img = img[:, :, n_sag // 2]
    elif corte == "Coronal":
        corte_idx = st.sidebar.slider("Selecciona el índice coronal", 0, n_cor - 1, n_cor // 2)
        coronal_img = img[:, corte_idx, :]
        axial_img = img[corte_idx, :, :]
        sagital_img = img[:, :, n_sag // 2]
    elif corte == "Sagital":
        corte_idx = st.sidebar.slider("Selecciona el índice sagital", 0, n_sag - 1, n_sag // 2)
        sagital_img = img[:, :, corte_idx]
        axial_img = img[corte_idx, :, :]
        coronal_img = img[:, n_cor // 2, :]

    def render2d(slice2d):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(apply_window_level(slice2d, ww, wc), cmap='gray', origin='lower')
        return fig

    rows = 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

    images_to_show = [
        axial_img,
        coronal_img,
        sagital_img,
        img[corte_idx, :, :]
    ]

    for i in range(4):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        ax.axis('off')
        ax.imshow(apply_window_level(images_to_show[i], ww, wc), cmap='gray', origin='lower')

    st.pyplot(fig)

    target_shape = (64, 64, 64)
    img_resized = resize(original_image, target_shape, anti_aliasing=True)
    x, y, z = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]

    st.sidebar.markdown('<p class="sub-header">Definir línea 3D con clicks</p>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    Instrucciones:
    - Haz click en la visualización 3D para establecer el punto A.
    - Haz un segundo click para establecer el punto B.
    - Para reiniciar, pulsa el botón 'Resetear puntos'.
    """, unsafe_allow_html=True)

    # Storage for points A and B
    if "points_selected" not in st.session_state:
        st.session_state.points_selected = []

    if st.sidebar.button("Resetear puntos"):
        st.session_state.points_selected = []

    # Build figure with volume
    fig3d = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=img_resized.flatten(),
        opacity=0.1,
        surface_count=15,
        colorscale="Gray",
    ))

    # If points defined add line and markers
    points = st.session_state.points_selected
    if len(points) > 0:
        xs, ys, zs = zip(*points)
        fig3d.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers+lines' if len(points) == 2 else 'markers',
            marker=dict(size=6, color='red'),
            line=dict(color='blue', width=4) if len(points) == 2 else None,
            name='Línea A-B'
        ))

    fig3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )

    # Capture click events using streamlit-plotly-events
    clicked_points = plotly_events(
        fig3d,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=600,
        key="3d_volume"
    )

    # When user clicks on 3D plot, record the coords as nearest voxel coords
    if clicked_points:
        click = clicked_points[0]  # Only one click at a time
        # click dict example: {'pointNumber': 0, 'curveNumber': 1, 'x': value, 'y': value, 'z': value}
        x_click, y_click, z_click = click['x'], click['y'], click['z']

        # Clip and round to integer voxel indices
        xi = int(np.clip(round(x_click), 0, target_shape[0] - 1))
        yi = int(np.clip(round(y_click), 0, target_shape[1] - 1))
        zi = int(np.clip(round(z_click), 0, target_shape[2] - 1))

        # Add point if less than 2 points selected
        if len(st.session_state.points_selected) < 2:
            st.session_state.points_selected.append((xi, yi, zi))
        else:
            # Replace second point if already two points selected
            st.session_state.points_selected[1] = (xi, yi, zi)

        # Rerun the app to pick up the change; this happens automatically on Streamlit interaction

    st.subheader("Vista 3D")
    st.plotly_chart(fig3d, use_container_width=True)

st.markdown('<p class="giant-title">Brachyanalysis</p>', unsafe_allow_html=True)
st.markdown("""
<hr>
<div style="text-align:center;color:#28aec5;font-size:14px;">
    Brachyanalysis - Visualizador de imágenes DICOM - con selección 3D por click
</div>
""", unsafe_allow_html=True)

