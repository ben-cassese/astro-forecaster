import pathlib
import setuptools


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setuptools.setup(
    name="astro-forecaster",
    version="2.0.2",
    author="Ben Cassese",
    author_email="b.c.cassese@columbia.edu",
    license="MIT",
    url="https://github.com/ben-cassese/astro-forecaster",
    description="Probabilistically forecast astronomical masses and radii",
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "scipy", "astropy", "h5py", "setuptools", "joblib"],
    packages=["forecaster"],
    include_package_data=True
)
