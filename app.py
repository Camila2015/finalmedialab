import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import tempfile
import random
import pyvista as pv
from streamlit_vtkjs import stpyvista

st.set_page_config(layout="wide")

st.title("Visualizador de Imágenes Médicas DICOM 2D y 3D con Anotaciones")

st.sidebar.header("Cargar archivos DICOM")
uploaded_files = st.sidebar.file_uploader("Sube múltiples archivos DICOM", type=["dcm"], accept_multiple_files=True)

if uploaded_files:
    # Crear directorio temporal para almacenar archivos
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepaths = []
        for uploaded_file in uploaded_files:
            filepath = os.path.join(tmpdirname, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            filepaths.append(filepath)

        # Leer los datos DICOM
        datasets = [pydicom.dcmread(fp) for fp in filepaths]
        datasets.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Ordenar por posición Z

        pixel_arrays = [ds.pixel_array for ds in datasets]
        volume = np.stack(pixel_arrays, axis=-1)

        st.sidebar.header("Controles de visualización 2D")
        axis = st.sidebar.radio("Selecciona eje", ["Axial (Z)", "Coronal (Y)", "Sagital (X)"])

        if axis == "Axial (Z)":
            index = st.sidebar.slider("Corte axial", 0, volume.shape[2] - 1, volume.shape[2] // 2)
            slice_2d = volume[:, :, index]
        elif axis == "Coronal (Y)":
            index = st.sidebar.slider("Corte coronal", 0, volume.shape[1] - 1, volume.shape[1] // 2)
            slice_2d = volume[:, index, :]
        else:
            index = st.sidebar.slider("Corte sagital", 0, volume.shape[0] - 1, volume.shape[0] // 2)
            slice_2d = volume[index, :, :]

        fig, ax = plt.subplots()
        ax.imshow(slice_2d, cmap="gray")
        ax.set_title(f"Corte {axis} en índice {index}")
        ax.axis("off")
        st.pyplot(fig)

        st.sidebar.header("Controles 3D y Anotaciones")

        if 'points' not in st.session_state:
            st.session_state['points'] = []
        if 'lines' not in st.session_state:
            st.session_state['lines'] = []
        if 'needles' not in st.session_state:
            st.session_state['needles'] = []

        add_point = st.sidebar.checkbox("Agregar punto 3D")
        x = st.sidebar.slider("X", 0, volume.shape[0]-1, volume.shape[0]//2)
        y = st.sidebar.slider("Y", 0, volume.shape[1]-1, volume.shape[1]//2)
        z = st.sidebar.slider("Z", 0, volume.shape[2]-1, volume.shape[2]//2)

        if add_point and st.sidebar.button("Agregar punto"):
            st.session_state['points'].append((x, y, z))

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Agregar líneas entre dos puntos")
        x1 = st.sidebar.slider("X1", 0, volume.shape[0]-1, volume.shape[0]//2, key="x1")
        y1 = st.sidebar.slider("Y1", 0, volume.shape[1]-1, volume.shape[1]//2, key="y1")
        z1 = st.sidebar.slider("Z1", 0, volume.shape[2]-1, volume.shape[2]//2, key="z1")
        x2 = st.sidebar.slider("X2", 0, volume.shape[0]-1, volume.shape[0]//2, key="x2")
        y2 = st.sidebar.slider("Y2", 0, volume.shape[1]-1, volume.shape[1]//2, key="y2")
        z2 = st.sidebar.slider("Z2", 0, volume.shape[2]-1, volume.shape[2]//2, key="z2")

        if st.sidebar.button("Agregar línea"):
            st.session_state['lines'].append(((x1, y1, z1), (x2, y2, z2)))

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Agujas")
        shape = st.sidebar.selectbox("Forma de la aguja", ["Recta", "Curva"])
        mode = st.sidebar.radio("Modo", ["Manual", "Aleatoria"])
        count = st.sidebar.number_input("Cantidad (si es aleatoria)", min_value=1, value=1)

        if st.button('Agregar aguja'):
            # Generar una o varias según modo
            times = count if mode == 'Aleatoria' else 1
            for _ in range(times):
                if mode == 'Aleatoria':
                    # Coordenadas fijas con z aleatorio
                    z_random = random.uniform(29, 36)
                    xa, ya, za = 32, 32, z_random
                    xb, yb, zb = 39, 32, z_random
                else:
                    xa, ya, za = x1, y1, z1
                    xb, yb, zb = x2, y2, z2
                pts = ((xa, ya, za), (xb, yb, zb))
                st.session_state['needles'].append({
                    'points': pts,
                    'color': f"#{random.randint(0,0xFFFFFF):06x}",
                    'curved': (shape == 'Curva')
                })

        # Crear el volumen en PyVista
        grid = pv.UniformGrid()
        grid.dimensions = np.array(volume.shape) + 1
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid.cell_data["values"] = volume.flatten(order="F")

        p = pv.Plotter()
        p.add_volume(grid, cmap="gray", opacity="sigmoid")

        for pt in st.session_state['points']:
            p.add_mesh(pv.Sphere(radius=1.0, center=pt), color="blue")

        for line in st.session_state['lines']:
            pts = np.array([line[0], line[1]])
            p.add_lines(pts, color="red", width=3)

        for needle in st.session_state['needles']:
            pts = np.array(needle['points'])
            if needle['curved']:
                mid = (pts[0] + pts[1]) / 2
                mid[2] += 3
                spline = pv.Spline([pts[0], mid, pts[1]], 100)
                p.add_mesh(spline, color=needle['color'], line_width=3)
            else:
                p.add_lines(pts, color=needle['color'], width=3)

        stpyvista(p, key="pv-render")
