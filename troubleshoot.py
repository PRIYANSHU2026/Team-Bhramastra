#!/usr/bin/env python3
"""
Troubleshooting script for the Point Cloud Processing application.
This script checks if all dependencies are correctly installed and
the environment is set up properly.
"""

import sys
import platform
import subprocess
import importlib
import os

def check_python_version():
    """Check if Python version is compatible"""
    print(f"Python version: {platform.python_version()}")
    if sys.version_info < (3, 6):
        print("❌ ERROR: Python 3.6 or newer is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_package(package_name):
    """Check if a package is installed and get its version"""
    try:
        package = importlib.import_module(package_name)
        if hasattr(package, '__version__'):
            version = package.__version__
        elif hasattr(package, 'version'):
            version = package.version
        else:
            version = "Unknown"
        print(f"✅ {package_name} is installed (version: {version})")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_qt():
    """Check for PyQt5 installation"""
    try:
        from PyQt5.QtCore import QT_VERSION_STR
        print(f"✅ PyQt5 is installed (Qt version: {QT_VERSION_STR})")
        return True
    except ImportError:
        print("❌ PyQt5 is not installed")
        return False

def check_open3d_version():
    """Check Open3D version and functionality"""
    try:
        import open3d as o3d
        print(f"✅ Open3D is installed (version: {o3d.__version__})")

        # Test basic Open3D functionality
        try:
            # Create a simple point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

            # Test normals
            pcd.estimate_normals()
            if pcd.has_normals():
                print("✅ Open3D normal estimation is working")
            else:
                print("❌ Open3D normal estimation failed")

            return True
        except Exception as e:
            print(f"❌ Open3D functionality test failed: {str(e)}")
            return False

    except ImportError:
        print("❌ Open3D is not installed")
        return False

def check_visualization():
    """Check visualization capabilities"""
    try:
        import open3d as o3d
        # Check if visualization module is available
        if hasattr(o3d, 'visualization'):
            print("✅ Open3D visualization module is available")
            return True
        else:
            print("❌ Open3D visualization module is not available")
            return False
    except ImportError:
        print("❌ Open3D is not installed")
        return False

def check_file_exists(filename):
    """Check if a file exists"""
    if os.path.exists(filename):
        print(f"✅ Found file: {filename}")
        return True
    else:
        print(f"❌ Missing file: {filename}")
        return False

def suggest_fixes(issues):
    """Suggest fixes for common issues"""
    if 'python' in issues:
        print("\nTo fix Python version issue:")
        print("1. Download and install Python 3.6 or newer from https://www.python.org/downloads/")
        print("2. Make sure the 'Add Python to PATH' option is checked during installation")

    if 'packages' in issues:
        print("\nTo fix missing packages:")
        print("1. Make sure you have activated your virtual environment (if using one)")
        print("2. Run: pip install -r requirements.txt")
        print("3. If that doesn't work, try installing packages individually:")
        print("   pip install open3d pyqt5 numpy imageio")

    if 'open3d' in issues:
        print("\nTo fix Open3D issues:")
        print("1. Try reinstalling Open3D: pip uninstall open3d && pip install open3d")
        print("2. Make sure you have compatible hardware for visualization")
        print("3. On some systems, you might need to install additional dependencies:")
        if platform.system() == "Linux":
            print("   sudo apt-get install libgl1-mesa-glx xvfb")
        elif platform.system() == "Darwin":  # macOS
            print("   brew install qt5 glfw")

    if 'files' in issues:
        print("\nTo fix missing files:")
        print("1. Make sure you are running this script from the project root directory")
        print("2. Re-download the project files if necessary")

def main():
    print("="*50)
    print("Point Cloud Processing App - Troubleshooting")
    print("="*50)

    issues = []

    # Check Python version
    if not check_python_version():
        issues.append('python')

    print("\n--- Checking required packages ---")
    packages_ok = True

    # Check core dependencies
    for package in ['numpy', 'imageio']:
        if not check_package(package):
            packages_ok = False

    if not check_qt():
        packages_ok = False

    if not check_open3d_version():
        packages_ok = False

    if not check_visualization():
        packages_ok = False

    if not packages_ok:
        issues.append('packages')

    # Check if Open3D has specific issues
    if 'open3d' in sys.modules:
        if not check_visualization():
            issues.append('open3d')

    print("\n--- Checking required files ---")
    files_ok = True
    for filename in ['point_cloud_app.py', 'requirements.txt']:
        if not check_file_exists(filename):
            files_ok = False

    if not files_ok:
        issues.append('files')

    # Summary
    print("\n" + "="*50)
    if not issues:
        print("✅ All checks passed! The environment appears to be properly set up.")
        print("If you're still experiencing issues, please check the error messages when running the application.")
    else:
        print("❌ Some issues were detected with your setup.")
        suggest_fixes(issues)

    print("="*50)

if __name__ == "__main__":
    main()
