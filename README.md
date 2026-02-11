# Examples accompanying the LibMobility paper

##  Usage:  

Start by cloning the repository:

```bash
$ git clone https://github.com/stochasticHydroTools/libmobility-paper-examples
```

### Getting the dependencies  
Dependencies are handled with [conda](https://github.com/conda-forge/miniforge), and can be installed with the following command in the root of the repository:

```bash
$ conda env create
$ conda activate libmobility-examples
```

### Running the examples

This repository contains a directory for each example showcased in the article.

## References
Fish, Ryker, Adam Carter, Pablo Diez-Silva, Rafael Delgado-Buscalioni, Raul P. Pelaez, and Brennan Sprinkle. “libMobility: A Python Library for Hydrodynamics at the Smoluchowski Level.” The Journal of Chemical Physics 164, no. 4 (2026): 044121. https://doi.org/10.1063/5.0304943.


## Contributing

- Work in a separated branch and do a pull request to `master` with your changes
- Add the required dependencies for your scripts to the environment file
- Create a new folder and place your scripts there, along with a README.md describing how to use them
