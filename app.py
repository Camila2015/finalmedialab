import streamlit as st
import numpy as np
import random
import plotly.graph_objects as go
import pydicom
import os
from skimage import measure
from scipy import ndimage

def dicom_to_mesh(dicom_folder, threshold=-300):
    files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm")]
    files.sort()
    slices = [pydicom.dcmread(f) for f in files]

    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = slices[0].SliceThickness
    spacing = map(float, (pixel_spacing[0], pixel_spacing[1], slice_thickness))
    spacing = np.array(list(spacing))

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    verts, faces, _, _ = measure.marching_cubes(image, level=threshold, spacing=spacing)
    return verts, faces

def visualize_3d(verts, faces, needles):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightblue',
            opacity=0.50
        )
    ])
    
    for needle in needles:
        fig.add_trace(go.Scatter3d(
            x=[needle[0][0], needle[1][0]],
            y=[needle[0][1], needle[1][1]],
            z=[needle[0][2], needle[1][2]],
            mode='lines+markers',
            marker=dict(size=3, color='red'),
            line=dict(color='red', width=5)
        ))

    fig.update_layout(scene=dict(aspectmode='data'))
    return fig

st.title("Visualización de Imágenes DICOM con Agujas 3D")

dicom_folder = st.text_input("Ruta de la carpeta con archivos DICOM:", "dicom_samples/")
if st.button("Cargar y Visualizar"):
    with st.spinner("Procesando imágenes DICOM..."):
        verts, faces = dicom_to_mesh(dicom_folder)

    st.success("Imágenes cargadas exitosamente.")

    st.subheader("Agregar Agujas")
    mode = st.radio("Modo de inserción:", ["Manual", "Aleatoria"])
    needles = []

    ORIGEN_AGUJA = (35, 57, 63)

    if mode == "Manual":
        st.write("Ingrese las coordenadas del segundo punto de la aguja (el primero es fijo en (35, 57, 63)):")
        x2 = st.slider("x2", 0, 100, 50)
        y2 = st.slider("y2", 0, 100, 50)
        z2 = st.slider("z2", 0, 100, 50)
        if st.button("Agregar Aguja Manual"):
            needles.append((ORIGEN_AGUJA, (x2, y2, z2)))
            st.success("Aguja agregada.")

    elif mode == "Aleatoria":
        num_needles = st.slider("Número de agujas aleatorias", 1, 10, 3)
        if st.button("Agregar Agujas Aleatorias"):
            for _ in range(num_needles):
                xb = random.uniform(30, 45)
                yb = random.uniform(30, 45)
                zb = random.uniform(30, 45)
                needles.append((ORIGEN_AGUJA, (xb, yb, zb)))
            st.success(f"{num_needles} agujas aleatorias agregadas.")

    if needles:
        fig = visualize_3d(verts, faces, needles)
        st.plotly_chart(fig, use_container_width=True)
