# Rollers
These scripts will recreate the rollers simulations. 

To generate simulation data, run the `rollers.py` file. Note that the parameters controlling gravitational height and stochasticity (`massFact` and `isDeterministic`) are located at the top of that file, and would need to be changed and rerun to generate the full dataset. Parameters used for the paper are `massFact = [0.1, 1.0]` and `isDeterministic = [True, False]`. Note that it is helpful to change the final simulation time depending on the case being run since low gravitational heights and deterministic simulations reach a characteristic time more quickly than other cases.

Once a dataset is generated, run the `process_data.py` file. This will compute the characteristic time for each run, then average the particle positions at the characteristic time across all runs, then save the averaged distributions.

Finally, the distribution plots can be recreated using the Matlab script `plotting/plot_dists.m`. The PlotOptix code to create the top-down roller visualizations is also included in `plotting/optix.py`, but note that it requires positions from one simulation at multiples of the characteristic time that can be produced by uncommenting a block of code within `process_data.py`.