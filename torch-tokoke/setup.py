from setuptools import setup, find_packages

setup(
    name="tokode-torch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",           # The core engine
        "torchdiffeq>=0.2.3",     # The ODE solvers
        "pandas>=2.0.0",          # Data management
        "numpy>=1.24.0",          # Math operations
        "pydantic>=2.0.0",        # Configuration registry
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0", "jupyter>=1.0.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0"]
    },
    python_requires=">=3.11",
    author="Tino Wells, Christopher Billingham, Phillip Prior",
    description="A PyTorch neural ODE framework for Tokamak plasma inductance dynamics.",
    url="https://github.com/caderache2014/ENGR521A",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)