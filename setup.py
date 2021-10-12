from setuptools import find_packages, setup

setup(
    name="WMAP-v5",
    version="1.0",
    description="WMAP likelihood for cobaya",
    author="Umberto Natale",
    author_email="NataleU@cardiff.ac.uk",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "cobaya>=3.1.0",
        "astropy",
    ],
    package_data={"wmap_v5": ["WMAP_v5.yaml"]},
)
