# Structure-preserving approximations of the Serre-Green-Naghdi equations in standard and hyperbolic form

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13120223.svg)](https://zenodo.org/doi/10.5281/zenodo.13120223)

This repository contains information and code to reproduce the results presented
in the article
```bibtex
@online{ranocha2024structure,
  title={Structure-preserving approximations of the {S}erre-{G}reen-{N}aghdi
         equations in standard and hyperbolic form},
  author={Ranocha, Hendrik and Ricchiuto, Mario},
  year={2024},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{ranocha2024structureRepro,
  title={Reproducibility repository for
         "{S}tructure-preserving approximations of the {S}erre-{G}reen-{N}aghdi
         equations in standard and hyperbolic form"},
  author={Ranocha, Hendrik and Ricchiuto, Mario},
  year={2024},
  howpublished={\url{https://github.com/ranocha/2024_serre_green_naghdi}},
  doi={10.5281/zenodo.13120223}
}
```

## Abstract

We develop structure-preserving numerical methods for the
Serre-Green-Naghdi equations, a model for weakly dispersive
free-surface waves. We consider both the classical form, 
embedding a non-linear elliptic operator,
and a hyperbolic approximation of the equations. Systems for 
both flat and variable topography are studied.
Our novel numerical methods conserve both the
total water mass and the total energy. In addition,
the methods for the original Serre-Green-Naghdi equations
conserve the total momentum for flat bathymetry.
For variable topography, all the methods proposed are well-balanced for the lake-at-rest state.
We provide  a theoretical setting allowing us to construct schemes
of any kind (finite difference, finite element, discontinuous Galerkin, spectral, etc.)
as long as summation-by-parts operators are available in the chosen setting.
Energy-stable variants are proposed by adding a consistent high-order artificial viscosity term.
The proposed methods are validated through a large set of benchmarks
to verify all the theoretical properties.
Whenever possible, comparisons with exact, reference numerical, or  experimental data are carried out.
The impressive advantage of structure preservation, and in particular energy preservation, to resolve accurately dispersive wave propagation
on very coarse meshes is demonstrated by several of the tests.


## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/).
The numerical experiments presented in this article were performed using
Julia v1.10.4.

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code` directory of this repository and follow the instructions
described in the `README.md` file therein.


## Authors

- [Hendrik Ranocha](https://ranocha.de) (Johannes Gutenberg University Mainz, Germany)
- [Mario Ricchiuto](https://team.inria.fr/cardamom/marioricchiuto) (Inria at University of Bordeaux, France)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
