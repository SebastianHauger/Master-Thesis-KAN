import dafi
from lorenz import lorenz
import yaml
import numpy as np
from trainODEKAN import get_data_lorenz63
import matplotlib.pyplot as plt


def get_vals():
    with open("lorenz.in", 'r') as f:
        input_lorenz = yaml.safe_load(f)
    t = np.arange(0, input_lorenz["t_end"], input_lorenz["dt_interval"])
    return t, input_lorenz
    


def plot_data(t, ts):
    fig, axarr = plt.subplots(ts.shape[1], 1, sharex=True)
    print(ts.shape)
    labels=["x", "y", "z"]    # maximum three labels
    for i,ax in enumerate(axarr):
        ax.plot(t, ts[:, i], color='black', lw=0.5)
        ax.set_ylabel(labels[i])
    ax.set_xlabel("time [s]")
    plt.show()
    
    

def gen_and_save_data():
    t, input_lorenz = get_vals()
    ts = lorenz(t, 
                init_state=np.array([input_lorenz["x_init"], input_lorenz["y_init"], input_lorenz["z_init"]]), 
                params=np.array([input_lorenz["rho"], input_lorenz["beta"], input_lorenz["sigma"]]))
    data = np.concatenate((t[:, np.newaxis], ts), axis=1)
    print(data.shape)
    np.savetxt("Results/results_lorenz/truth.dat", data, delimiter=' ', fmt='%f')
    

if __name__=='__main__':
    gen_and_save_data()
    ts, t = get_data_lorenz63()
    # print(ts[25000,:])
    plot_data(t, ts)
    
    
    

    # dafi.run('lorenz.py', "EnKF", nsamples=2, ntime=None,
    #          save_level='time', save_dir='Results/results_dafi',
    #          verbosity=1, inputs_model={"input_file": "lorenz.in"})
    