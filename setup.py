from setuptools import setup, find_packages

setup(
    name="point_cloud_processing",
    version="1.0.0",
    description="3D Point Cloud Processing Application with PyQt GUI",
    author="Point Cloud Processing Team",
    packages=find_packages(),
    install_requires=[
        "open3d>=0.15.0",
        "PyQt5>=5.15.0",
        "numpy>=1.19.0",
        "imageio>=2.21.0",
    ],
    entry_points={
        "console_scripts": [
            "point-cloud-app=point_cloud_app:main",
        ],
    },
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
