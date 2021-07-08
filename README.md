# Extract polynomial potentials from 3d magnetic fieldmaps.

author: Bernard Riemann, [ORCiD: 0000-0002-5102-9546](https://orcid.org/0000-0002-5102-9546)

used in:

B. Riemann and M. Aiba, "Algorithm to Analyze Complex Magnetic Structures Using a Tube Approach", Proc. IPAC21, Campinas, Brazil (2021).

## Requirements

The computations require Python and Scipy (see [requirements.txt](requirements.txt)). It is recommended to use a modern Linux distribution, although a setup in Windows should be possible in principle.

To execute the example notebook, it is recommended to also install [JupyterLab or Jupyter Notebook](https://jupyter.org).

## Installation

Install the ZyX module from the main directory

    sudo python setup.py install

## Usage

The demonstrations are in the `examples/` directory.

Generate the fieldmap files calling

    python example_fieldmaps.py

Then use the example Jupyter notebook [tubeApproach.ipynb](examples/TubeApproach.ipynb) to study the fieldmaps.
