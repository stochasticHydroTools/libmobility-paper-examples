import scattering_functions
import numpy as np
import json
import tqdm
import scattering_functions.msd

source_dirs = {
    'long':  'output/solver_NBody_N_122205_L_2560_t_final_28800_run_0/',
    'short': 'output/solver_NBody_N_122205_L_2560_t_final_3600_run_0/',
}
ts = {
    'long': 16,
    'short': 1,
}

for source in [
    'long',
    'short'
]:
    source_dir = source_dirs[source]
    colloids_filepath = f'{source_dir}/colloids.bin'
    binary_json_filepath = f'{source_dir}/binary_metadata.json'
    metadata_filepath = f'{source_dir}/params.json'

    with open(metadata_filepath) as metadata_file:
        metadata = json.load(metadata_file)
    with open(binary_json_filepath) as binary_json_file:
        binary_metadata = json.load(binary_json_file)


    particles_sim = np.fromfile(colloids_filepath, dtype=binary_metadata['dtype'])
    num_floats_per_row_sim = 1+metadata['N_colloids']*3
    assert particles_sim.size % num_floats_per_row_sim == 0, f'data_sim.size / num_floats_per_row_sim = {particles_sim.size / num_floats_per_row_sim}'
    particles_sim = particles_sim.reshape((-1, num_floats_per_row_sim))

    # reshape to trackpy format (needed by scattering_functions repo currently)
    num_expected_timesteps = particles_sim.shape[0]
    num_colloids = metadata['N_colloids']
    assert num_colloids == binary_metadata['N']

    print(f'creating data, will be {(num_colloids*num_expected_timesteps, 5)} {num_colloids * num_expected_timesteps * 5 * 4/1e9:.1f}GB')
    particles = np.full((num_colloids*num_expected_timesteps, 5), np.nan, dtype=np.float32)

    for t in tqdm.trange(num_expected_timesteps, desc='converting format simulation > Trackpy'):
        xyz_this_timestep = particles_sim[t, 1:]
        starting_row_i = t*num_colloids
        particles[starting_row_i:starting_row_i+num_colloids, [0, 1, 2]] = xyz_this_timestep.reshape((num_colloids, 3))
        particles[starting_row_i:starting_row_i+num_colloids, 3] = t
        particles[starting_row_i:starting_row_i+num_colloids, 4] = np.arange(num_colloids)

    # calculate MSD
    if source == 'short':
        msds, msds_unc = scattering_functions.msd.calc_incremental_xyz(particles, num_dimensions=3)
        msd_xy = msds[0, :] + msds[1, :]
    else:
        msd_xy = None


    # calculate intermediate scattering function f(k, t)

    particles_at_frame, times_at_frame = scattering_functions.get_particles_at_frame('F', particles, columns={
        'x': 0,
        'y': 1,
        't': 3
    })

    t = np.array([0, ts[source]])

    isf_results = scattering_functions.intermediate_scattering(
        F_type             = 'F',
        num_k_bins         = 60,
        # max_time_origins   = 1,
        max_time_origins   = 150,
        t                  = t,
        particles_at_frame = particles_at_frame,
        times_at_frame     = times_at_frame,
        max_k              = 21.816,
        min_k              = (2*np.pi/metadata['Lx'], 2*np.pi/metadata['Ly']),
        # cores            = 1 # set this if you have more cores available
    )

    t = t * metadata['t_save']

    np.savez(f'output/isf_{source}.npz', F=isf_results.F, k=isf_results.k, t=t,
             time_step=metadata['n_save'], msd=msd_xy, a=metadata['a'])
    print(f'saved output/isf_{source}.npz')