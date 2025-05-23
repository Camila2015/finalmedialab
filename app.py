import streamlit as st
import os
import pydicom
import numpy as np
import plotly.graph_objects as go
import tempfile
import zipfile

st.set_page_config(layout="wide", page_title="Visualizador 3D DICOM")

st.title("🧠 Visualizador 3D de archivos DICOM segmentados")

# 1. Cargar archivos
uploaded_file = st.file_uploader("Sube un archivo .zip con tus DICOMs", type="zip")

if uploaded_file:
    # 2. Extraer ZIP
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # 3. Leer los archivos DICOM
    dicom_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".dcm")]
    if not dicom_files:
        st.error("No se encontraron archivos DICOM en el .zip")
        st.stop()

    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices])
    volume = volume.astype(np.int16)
    volume = np.swapaxes(volume, 0, 2)  # Eje correcto para visualización

    # 4. Selección de umbral
    st.sidebar.title("🔧 Segmentación")
    threshold = st.sidebar.slider("Umbral de intensidad para visualizar", int(volume.min()), int(volume.max()), 300)

    # 5. Segmentar y visualizar
    x, y, z = volume.nonzero()
    values = volume[x, y, z]
    mask = values > threshold
    x, y, z = x[mask], y[mask], z[mask]
    values = values[mask]

    st.sidebar.write(f"Voxels visibles: {len(x)}")

    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=2, color=values, colorscale='Viridis', opacity=0.15)
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            bgcolor='black',
        ),
        paper_bgcolor='black',
        font_color='white',
        margin=dict(l=0, r=0, t=30, b=0),
        title="Volumen segmentado"
    )

    st.plotly_chart(fig, use_container_width=True)

