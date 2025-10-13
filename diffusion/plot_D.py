import matplotlib.pyplot as plt
import numpy as np

rescale_y = None

indexes_to_plot = {
    'short': np.index_exp[24:],
    'long': np.index_exp[:24],
}

for source in ['short', 'long']:
    data = np.load(f'isf_{source}.npz')

    F = data['F']
    k = data['k']
    t = data['t']
    print(t)

    if source == 'short':
        msd = data['msd']
        D_MSD = msd[1] / (4 * t[1])
        rescale_y = D_MSD


    # calculate f(k, t) = F(k, t) / F(k, 0)
    f = F / F[0, :]

    inversion_index = 1 # have to choose a time point to invert f(k, t) = exp(- D k^2 t) at
    D = - np.log(f[inversion_index, :]) / ( k**2 * t[inversion_index] )

    to_plot = indexes_to_plot[source]
    plt.scatter(k[to_plot], D[to_plot]/D_MSD, label=f'$t={t[inversion_index]:.0f}\mathrm{{s}}$')

plt.loglog()
plt.xlabel('$k$ ($\mathrm{\mu m^{-1}}$)')
plt.ylabel('$D(k)/D_0$')
plt.legend()
plt.ylim(0.8, 5)
plt.xlim()

plt.savefig('D_of_k.png')
print('saved D_of_k.png')