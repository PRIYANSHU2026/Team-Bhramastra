import sys
import os
import numpy as np
import open3d as o3d
import imageio.v2 as imageio
import tempfile
import webbrowser
import time
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                            QLabel, QFileDialog, QWidget, QComboBox, QGroupBox, QTabWidget,
                            QSplitter, QMessageBox, QProgressBar, QRadioButton, QButtonGroup,
                            QSlider, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt5.QtGui import QPixmap, QImage, QIcon, QKeyEvent, QMouseEvent, QWheelEvent, QCursor
from PyQt5.QtOpenGL import QGLWidget, QGLFormat

class PointCloudProcessingThread(QThread):
    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(object, object, object, object)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            # Load point cloud
            self.progress_signal.emit("Loading point cloud...", 10)
            pcd = self.load_point_cloud(self.file_path)

            # Orient normals
            self.progress_signal.emit("Orienting normals...", 30)
            pcd_with_normals = self.orient_normals(pcd)

            # Poisson reconstruction
            self.progress_signal.emit("Performing Poisson reconstruction...", 50)
            poisson_mesh = self.poisson_reconstruction(pcd_with_normals, depth=8)

            # Ball pivoting reconstruction
            self.progress_signal.emit("Performing Ball Pivoting reconstruction...", 80)
            ball_pivot_mesh = self.ball_pivoting_reconstruction(pcd_with_normals)

            self.progress_signal.emit("Processing complete!", 100)
            self.finished_signal.emit(pcd, pcd_with_normals, poisson_mesh, ball_pivot_mesh)

        except Exception as e:
            self.error_signal.emit(str(e))

    def load_point_cloud(self, file_path):
        """Load a point cloud from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist!")

        if file_path.endswith(('.xyz', '.pcd', '.ply', '.obj')):
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                raise ValueError(f"Failed to load {file_path}: no points found!")
        else:
            raise ValueError("Unsupported file format. Use .xyz, .pcd, .ply, or .obj")

        return pcd

    def orient_normals(self, pcd):
        """Orient normals consistently"""
        # Create a deep copy of the point cloud instead of using clone()
        pcd_copy = o3d.geometry.PointCloud()
        pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            pcd_copy.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))

        # Estimate and orient normals
        pcd_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_copy.orient_normals_consistent_tangent_plane(k=30)
        return pcd_copy

    def poisson_reconstruction(self, pcd, depth=8):
        """Perform Poisson surface reconstruction"""
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        mesh.compute_vertex_normals()
        colors = np.asarray([[0.0, 0.0, 1.0] for _ in range(len(mesh.vertices))])  # Blue
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    def ball_pivoting_reconstruction(self, pcd, radii=[1.0, 2.0, 4.0, 8.0, 16.0]):
        """Perform Ball Pivoting reconstruction"""
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances) if distances else 1.0
        radii = [r * avg_dist for r in radii]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.1, 0.3, 1.0])  # Bright blue color
        return mesh


class Open3DVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.geometry = None
        self.vis = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.rendering_active = False

        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Initialize movement parameters
        self.rotation_speed = 1.0
        self.translation_speed = 0.01
        self.zoom_speed = 0.05

        # Mouse tracking parameters
        self.setMouseTracking(True)
        self.last_mouse_pos = QPoint()
        self.mouse_pressed = False
        self.ctrl_pressed = False
        self.shift_pressed = False

        # Create a container for visualizer controls
        self.controls_container = QWidget()
        self.controls_layout = QHBoxLayout(self.controls_container)

        # Auto-rotate checkbox
        self.auto_rotate_checkbox = QCheckBox("Auto-Rotate")
        self.auto_rotate_checkbox.setChecked(False)
        self.auto_rotate_checkbox.toggled.connect(self.toggle_auto_rotate)
        self.controls_layout.addWidget(self.auto_rotate_checkbox)

        # Add color controls
        self.color_label = QLabel("Color:")
        self.controls_layout.addWidget(self.color_label)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["Default", "Blue", "Red", "Green", "Yellow", "White"])
        self.color_combo.currentIndexChanged.connect(self.change_color)
        self.controls_layout.addWidget(self.color_combo)

        # Add GIF export button
        self.export_gif_button = QPushButton("Export as GIF")
        self.export_gif_button.clicked.connect(self.export_as_gif)
        self.export_gif_button.setEnabled(False)
        self.controls_layout.addWidget(self.export_gif_button)

        # Add manual controls help button
        self.help_button = QPushButton("Keyboard & Mouse Controls")
        self.help_button.clicked.connect(self.show_controls)
        self.controls_layout.addWidget(self.help_button)

        # Add controls container to main layout
        self.layout.addWidget(self.controls_container)

        # Create a label to display "Loading..." initially
        self.loading_label = QLabel("No geometry loaded")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.loading_label)

        # Status label for controls
        self.status_label = QLabel("Click and drag to rotate, Shift+drag to pan, Mouse wheel to zoom")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

    # Mouse event handlers
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for rotation and panning"""
        if self.vis and self.rendering_active:
            self.mouse_pressed = True
            self.last_mouse_pos = event.pos()
            self.setFocus()  # Ensure widget has focus for keyboard events
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        self.mouse_pressed = False
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement for rotation and panning"""
        if not self.vis or not self.rendering_active or not self.mouse_pressed:
            super().mouseMoveEvent(event)
            return

        current_pos = event.pos()
        dx = current_pos.x() - self.last_mouse_pos.x()
        dy = current_pos.y() - self.last_mouse_pos.y()

        ctr = self.vis.get_view_control()

        # Pan with Shift key pressed or middle mouse button
        if event.modifiers() & Qt.ShiftModifier or event.buttons() & Qt.MiddleButton:
            # Scale dx and dy by translation speed
            ctr.translate(dx * self.translation_speed, dy * self.translation_speed)
        # Rotate with left mouse button
        elif event.buttons() & Qt.LeftButton:
            # Scale dx and dy by rotation speed
            ctr.rotate(dx * self.rotation_speed * 0.3, dy * self.rotation_speed * 0.3)

        self.last_mouse_pos = current_pos
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming"""
        if not self.vis or not self.rendering_active:
            super().wheelEvent(event)
            return

        # Get the number of degrees the wheel has turned
        delta = event.angleDelta().y()

        # Get view control for camera manipulation
        view_control = self.vis.get_view_control()

        try:
            if delta > 0:
                # Zoom in - Use a scale factor less than 1
                view_control.scale(0.9)
            else:
                # Zoom out - Use a scale factor greater than 1
                view_control.scale(1.1)

            # Update the UI
            self.status_label.setText(f"Zooming {'in' if delta > 0 else 'out'}")
            QTimer.singleShot(1500, lambda: self.status_label.setText(
                "Click and drag to rotate, Shift+drag to pan, Mouse wheel to zoom"))

        except Exception as e:
            # If scaling fails, show error
            self.status_label.setText(f"Zoom error: {str(e)}")
            print(f"Zoom error: {str(e)}")

        # Accept the event to prevent propagation
        event.accept()

    # Update help method to include mouse controls
    def show_controls(self):
        """Show a help dialog with keyboard and mouse controls"""
        controls_info = """
        3D Navigation Controls:

        Mouse Controls:
          • Click and drag: Rotate model
          • Shift+drag or Middle button drag: Pan model
          • Mouse wheel: Zoom in/out

        Keyboard Controls:
          Rotation:
            • Arrow Up/Down: Rotate up/down
            • Arrow Left/Right: Rotate left/right

          Translation (Pan):
            • W: Move up
            • S: Move down
            • A: Move left
            • D: Move right

          Zoom:
            • + or =: Zoom in
            • -: Zoom out

          Other:
            • R: Reset view

        * Click on the 3D view first to use controls
        """

        QMessageBox.information(self, "Navigation Controls", controls_info)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard interactions for manual maneuvering"""
        if not self.vis or not self.rendering_active:
            return

        ctr = self.vis.get_view_control()

        # Handle rotation with arrow keys
        if event.key() == Qt.Key_Up:
            ctr.rotate(0.0, -self.rotation_speed * 5)
        elif event.key() == Qt.Key_Down:
            ctr.rotate(0.0, self.rotation_speed * 5)
        elif event.key() == Qt.Key_Left:
            ctr.rotate(-self.rotation_speed * 5, 0.0)
        elif event.key() == Qt.Key_Right:
            ctr.rotate(self.rotation_speed * 5, 0.0)

        # Handle panning with WASD
        elif event.key() == Qt.Key_W:
            ctr.translate(0, -self.translation_speed)
        elif event.key() == Qt.Key_S:
            ctr.translate(0, self.translation_speed)
        elif event.key() == Qt.Key_A:
            ctr.translate(-self.translation_speed, 0)
        elif event.key() == Qt.Key_D:
            ctr.translate(self.translation_speed, 0)

        # Handle zooming with + and -
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:  # + key is often Shift+=
            ctr.scale(1 + self.zoom_speed)
        elif event.key() == Qt.Key_Minus:
            ctr.scale(1 - self.zoom_speed)

        # Reset view
        elif event.key() == Qt.Key_R:
            self.reset_view()

        super().keyPressEvent(event)

    def reset_view(self):
        """Reset the camera view to default position"""
        if self.vis:
            ctr = self.vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.8)

    def show_keyboard_controls(self):
        """Show a help dialog with keyboard controls"""
        controls_info = """
        Keyboard Controls for 3D Navigation:

        Rotation:
          • Arrow Up/Down: Rotate up/down
          • Arrow Left/Right: Rotate left/right

        Translation (Pan):
          • W: Move up
          • S: Move down
          • A: Move left
          • D: Move right

        Zoom:
          • + or =: Zoom in
          • -: Zoom out

        Other:
          • R: Reset view

        * Click on the 3D view first to use keyboard controls
        """

        QMessageBox.information(self, "Keyboard Controls", controls_info)

    def set_geometry(self, geometry):
        """Set the geometry to visualize"""
        self.geometry = geometry
        if self.geometry is not None:
            self.setup_visualization()
            self.export_gif_button.setEnabled(True)
        else:
            self.loading_label.setText("No geometry loaded")
            self.loading_label.show()
            self.export_gif_button.setEnabled(False)

    def setup_visualization(self):
        """Set up the Open3D visualization"""
        # Clean up any existing visualizer
        self.stop_visualization()

        # Hide the loading label
        self.loading_label.hide()

        # Center the geometry
        self.centered_geometry = self.center_geometry(self.geometry)

        try:
            # Create a new visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(visible=False)
            self.vis.add_geometry(self.centered_geometry)

            # Configure rendering options
            render_opt = self.vis.get_render_option()
            render_opt.background_color = np.array([0.1, 0.1, 0.1])
            render_opt.point_size = 3.0
            render_opt.light_on = True

            if isinstance(self.centered_geometry, o3d.geometry.TriangleMesh):
                self.centered_geometry.compute_vertex_normals()
                render_opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Default

            # Set up the camera view
            ctr = self.vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.8)

            # Start the visualization update timer
            self.rendering_active = True
            self.timer.start(50)  # Update every 50ms
        except Exception as e:
            QMessageBox.critical(None, "Visualization Error",
                                f"Failed to set up visualization: {str(e)}")
            self.loading_label.setText(f"Error: {str(e)}")
            self.loading_label.show()

    def toggle_auto_rotate(self, enabled):
        """Toggle automatic rotation"""
        self.auto_rotate = enabled

    def change_color(self, index):
        """Change the color of the geometry"""
        if self.geometry is None or self.centered_geometry is None:
            return

        color_map = {
            0: None,  # Default
            1: [0.0, 0.0, 1.0],  # Blue
            2: [1.0, 0.0, 0.0],  # Red
            3: [0.0, 1.0, 0.0],  # Green
            4: [1.0, 1.0, 0.0],  # Yellow
            5: [1.0, 1.0, 1.0],  # White
        }

        color = color_map.get(index)
        if color is None:
            return

        # Apply the color based on geometry type
        if isinstance(self.centered_geometry, o3d.geometry.TriangleMesh):
            self.centered_geometry.paint_uniform_color(color)
        elif isinstance(self.centered_geometry, o3d.geometry.PointCloud):
            self.centered_geometry.paint_uniform_color(color)

        # Update the visualizer
        if self.vis is not None:
            try:
                self.vis.update_geometry(self.centered_geometry)
            except Exception as e:
                print(f"Error updating geometry color: {str(e)}")

    def update_visualization(self):
        """Update the visualization"""
        if not self.rendering_active or self.vis is None:
            return

        try:
            # Rotate the view if auto-rotate is enabled
            if hasattr(self, 'auto_rotate') and self.auto_rotate:
                ctr = self.vis.get_view_control()
                ctr.rotate(1.0, 0.0)

            # Render the scene
            self.vis.poll_events()
            self.vis.update_renderer()

            # Capture the rendered image
            image = self.vis.capture_screen_float_buffer(False)
            if image is None:
                return

            # Convert to QImage and display
            image_np = np.asarray(image)
            height, width, _ = image_np.shape
            image_np = (image_np * 255).astype(np.uint8)
            image_np = np.ascontiguousarray(image_np)

            qimg = QImage(image_np.data, width, height,
                        width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # If we don't have a label yet, create one
            if not hasattr(self, 'image_label'):
                self.image_label = QLabel()
                self.image_label.setAlignment(Qt.AlignCenter)
                self.image_label.setMinimumSize(400, 300)
                self.layout.addWidget(self.image_label)

            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Error updating visualization: {str(e)}")
            self.timer.stop()
            self.rendering_active = False

    def export_as_gif(self):
        """Export the visualization as a rotating GIF"""
        if self.geometry is None:
            QMessageBox.warning(self, "Export Warning", "No geometry loaded to export")
            return

        # Get the save file path
        downloads_folder = self.get_downloads_folder()
        default_filename = "rotating_3d_view.gif"
        default_path = os.path.join(downloads_folder, default_filename)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save GIF", default_path, "GIF Files (*.gif)")

        if not file_path:
            return

        # Generate and save a rotating GIF
        try:
            # Store the current auto-rotate state
            original_auto_rotate = self.auto_rotate_checkbox.isChecked()

            # Create a progress dialog
            progress = QProgressBar()
            progress.setRange(0, 36)
            progress.setValue(0)

            progress_dialog = QMessageBox()
            progress_dialog.setWindowTitle("Creating GIF")
            progress_dialog.setText("Generating rotating GIF...")
            progress_dialog.setStandardButtons(QMessageBox.NoButton)

            layout = progress_dialog.layout()
            layout.addWidget(progress, 1, 0, 1, layout.columnCount())
            progress_dialog.show()

            # Update the UI
            QApplication.processEvents()

            with tempfile.TemporaryDirectory() as temp_dir:
                frame_paths = []

                # Save frames
                if self.vis is not None:
                    ctr = self.vis.get_view_control()

                    for i in range(36):
                        # Rotate and render
                        ctr.rotate(10.0, 0.0)
                        self.vis.poll_events()
                        self.vis.update_renderer()

                        # Capture frame
                        image_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
                        self.vis.capture_screen_image(image_path, do_render=True)
                        frame_paths.append(image_path)

                        # Update progress
                        progress.setValue(i + 1)
                        QApplication.processEvents()

                    # Create the GIF
                    images = [imageio.imread(path) for path in frame_paths]
                    imageio.mimsave(file_path, images, duration=0.1, loop=0)

            # Restore the original auto-rotate state
            self.auto_rotate_checkbox.setChecked(original_auto_rotate)

            # Close the progress dialog
            progress_dialog.accept()

            QMessageBox.information(self, "Export Successful",
                                    f"GIF saved to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                                f"Failed to create GIF: {str(e)}")

    def center_geometry(self, geometry):
        """Center geometry around origin"""
        if isinstance(geometry, o3d.geometry.PointCloud):
            points = np.asarray(geometry.points)
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            points = np.asarray(geometry.vertices)
        else:
            return geometry

        center = np.mean(points, axis=0)
        centered_points = points - center

        if isinstance(geometry, o3d.geometry.PointCloud):
            centered_geometry = o3d.geometry.PointCloud()
            centered_geometry.points = o3d.utility.Vector3dVector(centered_points)
            if geometry.has_normals():
                centered_geometry.normals = geometry.normals
            if geometry.has_colors():
                centered_geometry.colors = geometry.colors
        else:
            centered_geometry = o3d.geometry.TriangleMesh()
            centered_geometry.vertices = o3d.utility.Vector3dVector(centered_points)
            centered_geometry.triangles = geometry.triangles
            if geometry.has_vertex_normals():
                centered_geometry.vertex_normals = geometry.vertex_normals
            if geometry.has_vertex_colors():
                centered_geometry.vertex_colors = geometry.vertex_colors

        return centered_geometry

    def get_downloads_folder(self):
        """Get the path to the Downloads folder"""
        home = str(Path.home())
        downloads = os.path.join(home, 'Downloads')
        os.makedirs(downloads, exist_ok=True)
        return downloads

    def stop_visualization(self):
        """Stop the visualization"""
        self.timer.stop()
        self.rendering_active = False
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception as e:
                print(f"Error closing visualizer: {str(e)}")
            self.vis = None

    def closeEvent(self, event):
        """Clean up when the widget is closed"""
        self.stop_visualization()
        super().closeEvent(event)


class PointCloudProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Point Cloud Processing")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize geometry variables
        self.pcd = None
        self.pcd_with_normals = None
        self.poisson_mesh = None
        self.ball_pivot_mesh = None
        self.current_file_path = None

        # Set up the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create a horizontal layout for file selection
        self.file_selection_layout = QHBoxLayout()

        # Add file selection button
        self.file_button = QPushButton("Select Point Cloud File")
        self.file_button.clicked.connect(self.select_file)
        self.file_selection_layout.addWidget(self.file_button)

        # Add file path label
        self.file_label = QLabel("No file selected")
        self.file_selection_layout.addWidget(self.file_label)
        self.file_selection_layout.addStretch(1)

        # Add processing button
        self.process_button = QPushButton("Process Point Cloud")
        self.process_button.clicked.connect(self.process_point_cloud)
        self.process_button.setEnabled(False)
        self.file_selection_layout.addWidget(self.process_button)

        self.main_layout.addLayout(self.file_selection_layout)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        # Create a tab widget for different visualizations
        self.tab_widget = QTabWidget()

        # Create tabs for each visualization
        self.raw_pcd_tab = QWidget()
        self.pcd_normals_tab = QWidget()
        self.poisson_tab = QWidget()
        self.ball_pivot_tab = QWidget()
        self.comparison_tab = QWidget()

        # Add tabs to the tab widget
        self.tab_widget.addTab(self.raw_pcd_tab, "Raw Point Cloud")
        self.tab_widget.addTab(self.pcd_normals_tab, "Point Cloud with Normals")
        self.tab_widget.addTab(self.poisson_tab, "Poisson Mesh")
        self.tab_widget.addTab(self.ball_pivot_tab, "Ball Pivoting Mesh")
        self.tab_widget.addTab(self.comparison_tab, "Compare Meshes")

        # Set up each tab
        self.setup_raw_pcd_tab()
        self.setup_pcd_normals_tab()
        self.setup_poisson_tab()
        self.setup_ball_pivot_tab()
        self.setup_comparison_tab()

        self.main_layout.addWidget(self.tab_widget)

        # Create a layout for export options at the bottom
        self.export_layout = QHBoxLayout()

        # Create a group box for export options
        self.export_group = QGroupBox("Export Options")
        self.export_group_layout = QHBoxLayout()
        self.export_group.setLayout(self.export_group_layout)

        # Add a label
        self.export_label = QLabel("Export Format:")
        self.export_group_layout.addWidget(self.export_label)

        # Add format selection combobox
        self.format_combo = QComboBox()
        self.format_combo.addItems([".ply", ".obj", ".stl"])
        self.export_group_layout.addWidget(self.format_combo)

        # Add export button
        self.export_button = QPushButton("Export Selected Mesh")
        self.export_button.clicked.connect(self.export_mesh)
        self.export_button.setEnabled(False)
        self.export_group_layout.addWidget(self.export_button)

        self.export_layout.addWidget(self.export_group)

        # Create a group box for creators
        self.creators_group = QGroupBox("Project Details")
        self.creators_group_layout = QVBoxLayout()
        self.creators_group.setLayout(self.creators_group_layout)

        # Add creator information
        self.creators_info = QLabel(
            "Point Cloud Processing Tool\n"
            "Created for advanced 3D mesh reconstruction\n"
            "© 2025 Point Cloud Processing Team"
        )
        self.creators_info.setAlignment(Qt.AlignCenter)
        self.creators_group_layout.addWidget(self.creators_info)

        self.export_layout.addWidget(self.creators_group)

        self.main_layout.addLayout(self.export_layout)

    def setup_raw_pcd_tab(self):
        """Set up the raw point cloud tab"""
        layout = QVBoxLayout(self.raw_pcd_tab)
        self.raw_pcd_visualizer = Open3DVisualizer()
        layout.addWidget(self.raw_pcd_visualizer)

    def setup_pcd_normals_tab(self):
        """Set up the point cloud with normals tab"""
        layout = QVBoxLayout(self.pcd_normals_tab)
        self.pcd_normals_visualizer = Open3DVisualizer()
        layout.addWidget(self.pcd_normals_visualizer)

    def setup_poisson_tab(self):
        """Set up the Poisson mesh tab"""
        layout = QVBoxLayout(self.poisson_tab)
        self.poisson_visualizer = Open3DVisualizer()
        layout.addWidget(self.poisson_visualizer)

        # Add export controls specific to this tab
        export_layout = QHBoxLayout()
        export_button = QPushButton("Export Poisson Mesh")
        export_button.clicked.connect(lambda: self.export_specific_mesh('poisson'))
        export_layout.addWidget(export_button)
        layout.addLayout(export_layout)

    def setup_ball_pivot_tab(self):
        """Set up the Ball Pivoting mesh tab"""
        layout = QVBoxLayout(self.ball_pivot_tab)
        self.ball_pivot_visualizer = Open3DVisualizer()
        layout.addWidget(self.ball_pivot_visualizer)

        # Add export controls specific to this tab
        export_layout = QHBoxLayout()
        export_button = QPushButton("Export Ball Pivoting Mesh")
        export_button.clicked.connect(lambda: self.export_specific_mesh('ball_pivot'))
        export_layout.addWidget(export_button)
        layout.addLayout(export_layout)

    def setup_comparison_tab(self):
        """Set up the comparison tab with two visualizers side by side"""
        layout = QVBoxLayout(self.comparison_tab)

        # Create a splitter for the two visualizers
        splitter = QSplitter(Qt.Horizontal)

        # Left side container
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        # Left side selection
        left_selection = QGroupBox("Left View")
        left_selection_layout = QVBoxLayout()
        self.left_mesh_group = QButtonGroup()

        self.left_raw_radio = QRadioButton("Raw Point Cloud")
        self.left_normals_radio = QRadioButton("Point Cloud with Normals")
        self.left_poisson_radio = QRadioButton("Poisson Mesh")
        self.left_ball_pivot_radio = QRadioButton("Ball Pivoting Mesh")

        self.left_mesh_group.addButton(self.left_raw_radio, 1)
        self.left_mesh_group.addButton(self.left_normals_radio, 2)
        self.left_mesh_group.addButton(self.left_poisson_radio, 3)
        self.left_mesh_group.addButton(self.left_ball_pivot_radio, 4)

        self.left_poisson_radio.setChecked(True)

        left_selection_layout.addWidget(self.left_raw_radio)
        left_selection_layout.addWidget(self.left_normals_radio)
        left_selection_layout.addWidget(self.left_poisson_radio)
        left_selection_layout.addWidget(self.left_ball_pivot_radio)
        left_selection.setLayout(left_selection_layout)

        # Left side visualizer
        self.left_visualizer = Open3DVisualizer()

        left_layout.addWidget(left_selection)
        left_layout.addWidget(self.left_visualizer)

        # Right side container
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        # Right side selection
        right_selection = QGroupBox("Right View")
        right_selection_layout = QVBoxLayout()
        self.right_mesh_group = QButtonGroup()

        self.right_raw_radio = QRadioButton("Raw Point Cloud")
        self.right_normals_radio = QRadioButton("Point Cloud with Normals")
        self.right_poisson_radio = QRadioButton("Poisson Mesh")
        self.right_ball_pivot_radio = QRadioButton("Ball Pivoting Mesh")

        self.right_mesh_group.addButton(self.right_raw_radio, 1)
        self.right_mesh_group.addButton(self.right_normals_radio, 2)
        self.right_mesh_group.addButton(self.right_poisson_radio, 3)
        self.right_mesh_group.addButton(self.right_ball_pivot_radio, 4)

        self.right_ball_pivot_radio.setChecked(True)

        right_selection_layout.addWidget(self.right_raw_radio)
        right_selection_layout.addWidget(self.right_normals_radio)
        right_selection_layout.addWidget(self.right_poisson_radio)
        right_selection_layout.addWidget(self.right_ball_pivot_radio)
        right_selection.setLayout(right_selection_layout)

        # Right side visualizer
        self.right_visualizer = Open3DVisualizer()

        right_layout.addWidget(right_selection)
        right_layout.addWidget(self.right_visualizer)

        # Add containers to splitter
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)

        # Connect radio button changes to update visualizers
        self.left_mesh_group.buttonClicked.connect(self.update_comparison_views)
        self.right_mesh_group.buttonClicked.connect(self.update_comparison_views)

        layout.addWidget(splitter)

    def select_file(self):
        """Open a file dialog to select a point cloud file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Point Cloud File",
            "",
            "Point Cloud Files (*.ply *.pcd *.xyz *.obj)"
        )

        if file_path:
            self.current_file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.process_button.setEnabled(True)

    def process_point_cloud(self):
        """Process the selected point cloud file"""
        if not self.current_file_path:
            return

        # Disable the process button while processing
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)

        # Create and start the processing thread
        self.processing_thread = PointCloudProcessingThread(self.current_file_path)
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.error_signal.connect(self.processing_error)
        self.processing_thread.start()

    def update_progress(self, message, value):
        """Update the progress bar and status message"""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)

    def processing_finished(self, pcd, pcd_with_normals, poisson_mesh, ball_pivot_mesh):
        """Called when processing is complete"""
        self.pcd = pcd
        self.pcd_with_normals = pcd_with_normals
        self.poisson_mesh = poisson_mesh
        self.ball_pivot_mesh = ball_pivot_mesh

        # Set the geometries in the visualizers
        self.raw_pcd_visualizer.set_geometry(self.pcd)
        self.pcd_normals_visualizer.set_geometry(self.pcd_with_normals)
        self.poisson_visualizer.set_geometry(self.poisson_mesh)
        self.ball_pivot_visualizer.set_geometry(self.ball_pivot_mesh)

        # Update the comparison views
        self.update_comparison_views()

        # Enable export button
        self.export_button.setEnabled(True)

        # Hide progress bar and re-enable process button
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)

        self.statusBar().showMessage("Processing complete!")

    def processing_error(self, error_message):
        """Called when an error occurs during processing"""
        QMessageBox.critical(self, "Processing Error", error_message)
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        self.statusBar().showMessage("Error during processing")

    def update_comparison_views(self):
        """Update the comparison view visualizers based on radio button selection"""
        if not all([self.pcd, self.pcd_with_normals, self.poisson_mesh, self.ball_pivot_mesh]):
            return

        # Get the selected options
        left_id = self.left_mesh_group.checkedId()
        right_id = self.right_mesh_group.checkedId()

        # Update left visualizer
        if left_id == 1:
            self.left_visualizer.set_geometry(self.pcd)
        elif left_id == 2:
            self.left_visualizer.set_geometry(self.pcd_with_normals)
        elif left_id == 3:
            self.left_visualizer.set_geometry(self.poisson_mesh)
        elif left_id == 4:
            self.left_visualizer.set_geometry(self.ball_pivot_mesh)

        # Update right visualizer
        if right_id == 1:
            self.right_visualizer.set_geometry(self.pcd)
        elif right_id == 2:
            self.right_visualizer.set_geometry(self.pcd_with_normals)
        elif right_id == 3:
            self.right_visualizer.set_geometry(self.poisson_mesh)
        elif right_id == 4:
            self.right_visualizer.set_geometry(self.ball_pivot_mesh)

    def export_mesh(self):
        """Export the currently selected mesh"""
        # Determine which tab is currently active
        current_tab = self.tab_widget.currentWidget()

        if current_tab == self.poisson_tab:
            self.export_specific_mesh('poisson')
        elif current_tab == self.ball_pivot_tab:
            self.export_specific_mesh('ball_pivot')
        elif current_tab == self.comparison_tab:
            # Check which mesh is selected in the right view (arbitrarily choosing right)
            right_id = self.right_mesh_group.checkedId()
            if right_id == 3:
                self.export_specific_mesh('poisson')
            elif right_id == 4:
                self.export_specific_mesh('ball_pivot')
            else:
                QMessageBox.information(self, "Export Info", "Please select a mesh (not a point cloud) to export")
        else:
            QMessageBox.information(self, "Export Info", "Please switch to a tab with a mesh to export")

    def export_specific_mesh(self, mesh_type):
        """Export a specific mesh type"""
        if mesh_type == 'poisson' and self.poisson_mesh is None:
            QMessageBox.warning(self, "Export Error", "No Poisson mesh available to export")
            return

        if mesh_type == 'ball_pivot' and self.ball_pivot_mesh is None:
            QMessageBox.warning(self, "Export Error", "No Ball Pivoting mesh available to export")
            return

        # Get the selected format
        selected_format = self.format_combo.currentText()

        # Get the downloads folder
        downloads_folder = self.get_downloads_folder()

        # Set default filename based on original file and mesh type
        original_filename = os.path.splitext(os.path.basename(self.current_file_path))[0]
        default_filename = f"{original_filename}_{mesh_type}{selected_format}"
        default_path = os.path.join(downloads_folder, default_filename)

        # Open save dialog
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Mesh",
            default_path,
            f"Mesh Files (*{selected_format})"
        )

        if not save_path:
            return

        try:
            if mesh_type == 'poisson':
                o3d.io.write_triangle_mesh(save_path, self.poisson_mesh)
            elif mesh_type == 'ball_pivot':
                o3d.io.write_triangle_mesh(save_path, self.ball_pivot_mesh)

            QMessageBox.information(self, "Export Successful", f"Mesh saved to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save mesh: {str(e)}")

    def get_downloads_folder(self):
        """Get the path to the Downloads folder on Mac"""
        home = str(Path.home())
        downloads = os.path.join(home, 'Downloads')
        os.makedirs(downloads, exist_ok=True)
        return downloads

    def closeEvent(self, event):
        """Clean up when the application is closed"""
        # Stop all visualizers
        try:
            self.raw_pcd_visualizer.stop_visualization()
            self.pcd_normals_visualizer.stop_visualization()
            self.poisson_visualizer.stop_visualization()
            self.ball_pivot_visualizer.stop_visualization()
            self.left_visualizer.stop_visualization()
            self.right_visualizer.stop_visualization()
        except:
            pass

        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = PointCloudProcessingApp()

    # Check for command line arguments (file path)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        file_path = sys.argv[1]
        # Set the file path
        window.current_file_path = file_path
        window.file_label.setText(os.path.basename(file_path))
        window.process_button.setEnabled(True)
        # Optional: auto-process the file
        QTimer.singleShot(500, window.process_point_cloud)

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
