# RAIL sompz

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/223043497.svg)](https://zenodo.org/badge/latestdoi/223043497)
[![codecov](https://codecov.io/gh/LSSTDESC/rail_sompz/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/rail_sompz)
[![PyPI](https://img.shields.io/pypi/v/pz-rail-sompz?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/pz-rail-sompz/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSSTDESC/rail_sompz/smoke-test.yml)](https://github.com/LSSTDESC/rail_sompz/actions/workflows/smoke-test.yml)

**sompz** - RAIL estimator, summarizer, and classifier using the SOMPZ method described in [Buchs, Davis, et al. 2019](https://arxiv.org/pdf/1901.05005.pdf), [Sánchez, Raveri, Alarcon, Bernstein 2020](https://arxiv.org/pdf/2004.09542.pdf), [Myles, Alarcon et al. 2021](https://arxiv.org/pdf/2012.08566.pdf) and [Campos, et al. 2023 (In preparation)](https://github.com/AndresaCampos/sompz_y6). 

# RAIL: Redshift Assessment Infrastructure Layers

RAIL is a flexible software library providing tools to produce at-scale
photometric redshift data products, including uncertainties and summary
statistics, and stress-test them under realistically complex systematics.
A detailed description of RAIL's modular structure is available in the 
[Overview](https://lsstdescrail.readthedocs.io/en/latest/source/overview.html) 
on ReadTheDocs.

RAIL serves as the infrastructure supporting many extragalactic applications 
of the Legacy Survey of Space and Time (LSST) on the Vera C. Rubin Observatory,
including Rubin-wide commissioning activities. RAIL was initiated by the
Photometric Redshifts (PZ) Working Group (WG) of the LSST Dark Energy Science 
Collaboration (DESC) as a result of the lessons learned from the 
[Data Challenge 1 (DC1) experiment](https://academic.oup.com/mnras/article/499/2/1587/5905416) 
to enable the PZ WG Deliverables in 
[the LSST-DESC Science Roadmap (see Sec. 5.18)](https://lsstdesc.org/assets/pdf/docs/DESC_SRM_latest.pdf), 
aiming to guide the selection and implementation of redshift estimators in DESC
analysis pipelines. RAIL is developed and maintained by a diverse team
comprising DESC Pipeline Scientists (PSs), international in-kind contributors,
LSST Interdisciplinary Collaboration for Computing (LINCC) Frameworks software
engineers, and other volunteers, but all are welcome to join the team
regardless of LSST data rights. 

## Installation

Installation instructions are available under 
[Installation](https://lsstdescrail.readthedocs.io/en/latest/source/installation.html)
on ReadTheDocs.

## Contributing

The greatest strength of RAIL is its extensibility; those interested in
contributing to RAIL should start by consulting the 
[Contributing guidelines](https://lsstdescrail.readthedocs.io/en/latest/source/contributing.html)
on ReadTheDocs.

## Citing RAIL

RAIL is open source and may be used according to the terms of its 
[LICENSE](https://github.com/LSSTDESC/RAIL/blob/main/LICENSE) 
[(BSD 3-Clause)](https://opensource.org/licenses/BSD-3-Clause).
If you make use of the ideas or software here in any publication, you must cite
this repository <https://github.com/LSSTDESC/RAIL> as "LSST-DESC PZ WG (in prep)" 
with the [Zenodo DOI](https://doi.org/10.5281/zenodo.7017551).
Please consider also inviting the developers as co-authors on publications
resulting from your use of RAIL by 
[making an issue](https://github.com/LSSTDESC/RAIL/issues/new/choose).
Additionally, several of the codes accessible through the RAIL ecosystem must 
be cited if used in a publication. A convenient list of what to cite may be found under 
[Citing RAIL](https://lsstdescrail.readthedocs.io/en/latest/source/citing.html) on ReadTheDocs.
