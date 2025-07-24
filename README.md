# Examples accompanying the LibMobility paper

##  Usage:  

The project can be built and installed via CMake, or pip if the Python wrapper is needed.

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

This repository contains a directory for each example showcased in the article, namely:

- `benchmarks/`: Contains a script that will gather some performance metrics and visualize them.


## References

TODO: [INSERT PAPER REFERENCE WHEN WE HAVE A DOI]


## Contributing

- Add the required dependencies for your scripts to the environment file
- Create a new folder and place your scripts there, along with a README.md describing how to use them
