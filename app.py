import streamlit as st
import os
import tempfile
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import themes
from stpyvista import stpyvista

st.set_page_config(layout="wide")
st.title("Visualizador DICOM 2D/3D con Anotaciones")

# Configurar el tema de PyVista
plot_theme = themes.DefaultTheme()
plot_theme.background = 'white'
plot_theme.color = 'black'
plot_theme.font.family = 'arial'

# Funciones

def load_dicom_series(files):
    datasets = [pydicom.dcmread(f) for f in files]
    datasets.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    slices = [ds.pixel_array for ds in datasets]
    img3d = np.stack(slices, axis=0)
    img3d = img3d.astype(np.int16)
    return img3d, datasets[0]

def apply_window(img, window_width, window_center):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(img, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    return windowed

def plot_3d_volume(img3d):
    volume = pv.wrap(img3d)
    volume.spacing = (1, 1, 1)
    return volume

def draw_line(p1, p2):
    line = pv.Line(p1, p2)
    return line

def draw_spline(points):
    return pv.Spline(points, 1000)

# Subir archivos DICOM
files = st.file_uploader("Sube archivos DICOM", type=['dcm'], accept_multiple_files=True)

if files:
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for file in files:
        path = os.path.join(temp_dir, file.name)
        with open(path, 'wb') as f:
            f.write(file.read())
        file_paths.append(path)
    
    img, meta = load_dicom_series(file_paths)
    st.sidebar.markdown(f"**Dimensiones del volumen:** {img.shape}")

    # Mostrar imágenes 2D con coordenadas clickeadas
    if img is not None:
        n_ax, n_cor, n_sag = img.shape
        mn, mx = float(img.min()), float(img.max())
        default = {'ww': mx - mn, 'wc': mn + (mx - mn)/2}

        sync = st.sidebar.checkbox('Sincronizar cortes', True)
        if sync:
            orientation = st.sidebar.radio('Corte', ['Axial', 'Coronal', 'Sagital'])
            limits = {'Axial': n_ax, 'Coronal': n_cor, 'Sagital': n_sag}
            idx = st.sidebar.slider('Índice', 0, limits[orientation]-1, limits[orientation]//2)
        else:
            orientation = st.sidebar.selectbox('Corte', ['Axial', 'Coronal', 'Sagital'])
            idx = st.sidebar.slider('Índice', 0, img.shape[['Axial', 'Coronal', 'Sagital'].index(orientation)]-1,
                                     img.shape[['Axial', 'Coronal', 'Sagital'].index(orientation)]//2)
        invert = st.sidebar.checkbox('Negativo', False)
        wtype = st.sidebar.selectbox('Tipo ventana', ['Default', 'Abdomen', 'Hueso', 'Pulmón'])
        presets = {'Abdomen': (400, 40), 'Hueso': (2000, 500), 'Pulmón': (1500, -600)}
        ww, wc = presets.get(wtype, (default['ww'], default['wc']))

        slices = {
            'Axial': img[idx, :, :],
            'Coronal': img[:, idx, :],
            'Sagital': img[:, :, idx]
        }

        if 'click_coords' not in st.session_state:
            st.session_state.click_coords = {'Axial': None, 'Coronal': None, 'Sagital': None}

        cols = st.columns(3)
        for col, (name, data) in zip(cols, slices.items()):
            with col:
                st.markdown(name)
                fig, ax = plt.subplots()
                ax.axis('off')
                img2d = apply_window(data, ww, wc)
                if invert:
                    img2d = 1 - img2d
                im = ax.imshow(img2d, cmap='gray', origin='lower')

                def onclick(event):
                    if event.xdata is not None and event.ydata is not None:
                        x = int(event.xdata)
                        y = int(event.ydata)
                        st.session_state.click_coords[name] = (x, y)

                fig.canvas.mpl_connect('button_press_event', onclick)
                st.pyplot(fig)

                if st.session_state.click_coords[name]:
                    x, y = st.session_state.click_coords[name]
                    if name == 'Axial':
                        st.info(f'Coordenadas Axial: X={x}, Y={y}, Z={idx}')
                    elif name == 'Coronal':
                        st.info(f'Coordenadas Coronal: X={x}, Y={idx}, Z={y}')
                    elif name == 'Sagital':
                        st.info(f'Coordenadas Sagital: X={idx}, Y={x}, Z={y}')

    # Render 3D
    st.subheader("Visualización 3D")
    volume = plot_3d_volume(img)
    p = pv.Plotter(theme=plot_theme, notebook=False, off_screen=True)
    p.add_volume(volume, cmap="gray", opacity="sigmoid")

    # Añadir anotaciones
    st.sidebar.subheader("Anotaciones 3D")
    annotation_mode = st.sidebar.selectbox("Tipo de anotación", ["Ninguna", "Punto", "Línea recta", "Línea curva"])

    if annotation_mode != "Ninguna":
        if 'annotations' not in st.session_state:
            st.session_state.annotations = []

        x = st.sidebar.number_input("X", min_value=0, max_value=img.shape[2]-1)
        y = st.sidebar.number_input("Y", min_value=0, max_value=img.shape[1]-1)
        z = st.sidebar.number_input("Z", min_value=0, max_value=img.shape[0]-1)
        if st.sidebar.button("Agregar punto"):
            st.session_state.annotations.append((x, y, z))

        for pt in st.session_state.annotations:
            p.add_mesh(pv.Sphere(center=pt, radius=1), color='red')

        if annotation_mode == "Línea recta" and len(st.session_state.annotations) >= 2:
            line = draw_line(st.session_state.annotations[-2], st.session_state.annotations[-1])
            p.add_mesh(line, color='blue', line_width=3)

        elif annotation_mode == "Línea curva" and len(st.session_state.annotations) >= 2:
            spline = draw_spline(st.session_state.annotations)
            p.add_mesh(spline, color='green', line_width=3)

    stpyvista(p, key="3d")

    # Pie de página
    st.markdown("---")
    st.caption("App desarrollada para visualización y anotación de imágenes médicas DICOM.")
