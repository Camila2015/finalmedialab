import streamlit as st
import os
import pydicom
import numpy as np
import pyvista as pv
import tempfile
import random

# Inicializar sesión
if 'points' not in st.session_state:
    st.session_state['points'] = []
if 'lines' not in st.session_state:
    st.session_state['lines'] = []
if 'needles' not in st.session_state:
    st.session_state['needles'] = []

st.title("Visor de Imágenes Médicas DICOM 2D/3D")

uploaded_files = st.file_uploader("Sube archivos DICOM", accept_multiple_files=True)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdirname:
        paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(tmpdirname, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            paths.append(path)

        datasets = [pydicom.dcmread(p) for p in paths]
        datasets.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        try:
            volume = np.stack([ds.pixel_array for ds in datasets])
            volume = volume.astype(np.int16)

            spacing = list(map(float, datasets[0].PixelSpacing))
            spacing.append(float(datasets[0].SliceThickness))
            spacing = spacing[::-1]
            vol_shape = volume.shape

            grid = pv.UniformGrid()
            grid.dimensions = np.array(vol_shape) + 1
            grid.origin = (0, 0, 0)
            grid.spacing = spacing
            grid.cell_data["values"] = volume.flatten(order="F")
        except Exception as e:
            st.error(f"Error procesando los archivos DICOM: {e}")
            st.stop()

        view = st.radio("Selecciona la vista", ["2D", "3D"])

        if view == "2D":
            axis = st.radio("Eje", ["Sagital", "Coronal", "Axial"])
            slice_idx = st.slider("Índice de la imagen", 0, volume.shape[0]-1 if axis == "Axial" else volume.shape[1]-1 if axis == "Coronal" else volume.shape[2]-1)

            if axis == "Axial":
                img = volume[slice_idx, :, :]
            elif axis == "Coronal":
                img = volume[:, slice_idx, :]
            elif axis == "Sagital":
                img = volume[:, :, slice_idx]

            st.image(img, caption=f"Vista {axis} - Corte {slice_idx}", clamp=True)

        else:
            plotter = pv.Plotter(off_screen=True, window_size=[800,800])
            opacity = st.slider("Opacidad del volumen", 0.0, 1.0, 0.15)
            plotter.add_volume(grid, opacity=opacity, cmap="bone")

            for i, pt in enumerate(st.session_state['points']):
                plotter.add_mesh(pv.Sphere(radius=1.0, center=pt), color="red")
                plotter.add_point_labels([pt], [str(i)], point_size=20, font_size=36)

            for ln in st.session_state['lines']:
                line = pv.Line(ln[0], ln[1])
                plotter.add_mesh(line, color="green", line_width=5)

            for needle in st.session_state['needles']:
                p1, p2 = needle['points']
                if needle['curved']:
                    mid = [(p1[i]+p2[i])/2 for i in range(3)]
                    mid[1] += 5
                    spline = pv.Spline([p1, mid, p2], 50)
                    plotter.add_mesh(spline, color=needle['color'], line_width=3)
                else:
                    line = pv.Line(p1, p2)
                    plotter.add_mesh(line, color=needle['color'], line_width=3)

            plotter.show(jupyter_backend='pythreejs', screenshot='output.png')
            st.image('output.png')

        st.sidebar.title("Anotaciones 3D")

        with st.sidebar.expander("Agregar Punto 3D"):
            x = st.number_input("X", value=0.0)
            y = st.number_input("Y", value=0.0)
            z = st.number_input("Z", value=0.0)
            if st.button("Agregar punto"):
                st.session_state['points'].append((x, y, z))

        with st.sidebar.expander("Dibujar Línea"):
            if len(st.session_state['points']) >= 2:
                idx1 = st.number_input("Índice punto 1", 0, len(st.session_state['points'])-1)
                idx2 = st.number_input("Índice punto 2", 0, len(st.session_state['points'])-1)
                if st.button("Dibujar línea"):
                    p1 = st.session_state['points'][idx1]
                    p2 = st.session_state['points'][idx2]
                    st.session_state['lines'].append((p1, p2))
            else:
                st.info("Agrega al menos 2 puntos primero.")

        with st.sidebar.expander("Agregar Agujas"):
            mode = st.radio("Modo", ["Manual", "Aleatoria"])
            shape = st.radio("Forma", ["Recta", "Curva"])
            count = st.slider("Cantidad (aleatorio)", 1, 10, 3) if mode == "Aleatoria" else 1
            x1 = st.number_input("x1", 0, 100, 10)
            y1 = st.number_input("y1", 0, 100, 10)
            z1 = st.number_input("z1", 0, 100, 10)
            x2 = st.number_input("x2", 0, 100, 20)
            y2 = st.number_input("y2", 0, 100, 20)
            z2 = st.number_input("z2", 0, 100, 20)

            if st.button('Agregar aguja'):
                times = count if mode == 'Aleatoria' else 1
                for _ in range(times):
                    if mode == 'Aleatoria':
                        z_val = random.randint(29, 36)
                        xa, ya, za = 32, 32, z_val
                        xb, yb, zb = 39, 32, z_val
                    pts = ((x1,y1,z1),(x2,y2,z2)) if mode == 'Manual' else ((xa,ya,za),(xb,yb,zb))
                    st.session_state['needles'].append({
                        'points': pts,
                        'color': f"#{random.randint(0,0xFFFFFF):06x}",
                        'curved': (shape == 'Curva')
                    })

        if st.sidebar.button("Limpiar Todo"):
            st.session_state['points'] = []
            st.session_state['lines'] = []
            st.session_state['needles'] = []
