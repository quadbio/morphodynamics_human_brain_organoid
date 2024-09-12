from setuptools import setup, find_packages
from pathlib import Path
project_root_dir = Path(__file__).parent.resolve()
with open(project_root_dir / "requirements.txt", "r", encoding="utf-8") as _file:
    INSTALL_REQUIRES = _file.read().splitlines()

setup(
    name='phenoscapes',
    version='0.1.1',
    url='https://github.com/quadbio/morphodynamics_human_brain_organoid/4i_processing',
    license='MIT',
    author='Christoph Harmel',
    author_email='christoph.harmel@roche.com',
    description='framework for processing 4i data.',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    entry_points={'console_scripts': ['phenoscapes=phenoscapes.cli:main']}
)
