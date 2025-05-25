import streamlit as st
import pydicom
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
from skimage.transform import resize
from skimage import measure

st.set_page_config(layout="wide")
st.title("Visualizador DICOM 2D y 3D (Compatible con Streamlit Cloud)")

uploaded_files = st.file_uploader("Sube los archivos DICOM", type=["dcm"], accept_multiple_files=True)

@st.cache_data(show_spinner=False)
def load_dicom_series(files):
    slices = [pydicom.dcmread(f) for f in files]
    slices = [s for s in slices if hasattr(s, 'InstanceNumber')]
    slices.sort(key=lambda s: s.InstanceNumber)
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = getattr(slices[0], 'SliceThickness', 1.0)
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    spacing = list(map(float, pixel_spacing)) + [float(slice_thickness)]
    return image, spacing

@st.cache_data(show_spinner=False)
def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def show_slice(slice_img, axis):
    st.image(slice_img, caption=f"Corte {axis}", use_column_width=True, clamp=True)

def create_3d_plot(volume):
    vol = normalize_image(volume)
    vol = resize(vol, (64, 64, 64), mode='constant')
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    x, y, z = verts.T
    i, j, k = faces.T
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightblue', opacity=0.5)
    layout = go.Layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=[mesh], layout=layout)
    return fig

if uploaded_files:
    with st.spinner("Cargando datos DICOM..."):
        volume, spacing = load_dicom_series(uploaded_files)
        volume = normalize_image(volume)

    axial = volume[volume.shape[0] // 2, :, :]
    coronal = volume[:, volume.shape[1] // 2, :]
    sagittal = volume[:, :, volume.shape[2] // 2]

    col1, col2, col3 = st.columns(3)
    with col1:
        show_slice(axial, "Axial")
    with col2:
        show_slice(coronal, "Coronal")
    with col3:
        show_slice(sagittal, "Sagital")

    st.subheader("Visualización 3D")
    fig = create_3d_plot(volume)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sube varios archivos DICOM para comenzar.")
