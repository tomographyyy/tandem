from setuptools import setup, find_packages

def load_requires_from_file(filepath):
    with open(filepath) as fp:
        return [pkg_name.strip() for pkg_name in fp.readlines()]

setup(
    name='tandem',
    version="0.0.1",
    description="Tsunami Adjoint Simulator",
    author='Tomohiro TAKAGAWA',
    packages=find_packages(),
    license='Apache-2.0',
    install_requires=load_requires_from_file("requirements.txt")
)
