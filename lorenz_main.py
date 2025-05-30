import dafi
from scipy.integrate import ode
import yaml
import numpy as np
from trainODEKAN import get_data_lorenz63
import matplotlib.pyplot as plt


def lorenz(time_series, init_state, params):
    def lorenz63(time, state, params):
        x, y, z = state
        rho, beta, sigma = params
        dx = sigma * (y - x)
        dy = rho*x - y - x*z
        dz = x*y - beta*z
        return [dx, dy, dz]
    solver = ode(lorenz63)
    solver.set_integrator('dopri5') 
    solver.set_initial_value(init_state, time_series[0])
    solver.set_f_params(params)
    state = np.expand_dims(np.array(init_state), axis=0)
    for time in time_series[1:]:
        if not solver.successful():
            raise RuntimeError(f"solver failed at time {time}")
        else:
            solver.integrate(time)
            
        state = np.vstack((state, solver.y))
    return state 


def get_vals():
    with open("lorenz.in", 'r') as f:
        input_lorenz = yaml.safe_load(f)
    t = np.arange(0, input_lorenz["t_end"], input_lorenz["dt_interval"])
    return t, input_lorenz
    


def plot_data(t, ts, name):
    fig, axarr = plt.subplots(ts.shape[1], 1, sharex=True)
    print(ts.shape)
    labels=["x", "y", "z"]    # maximum three labels
    for i,ax in enumerate(axarr):
        ax.plot(t, ts[:, i], color='black', lw=0.5)
        ax.set_ylabel(labels[i])
    ax.set_xlabel("time [s]")
    fig.suptitle(name)
    plt.savefig('images/lorenz/ts'+name+".pdf", dpi=200)
    plt.show()
    
    

def gen_and_save_data(filename):
    t, input_lorenz = get_vals()
    ts = lorenz(t, 
                init_state=np.array([input_lorenz["x_init"], input_lorenz["y_init"], input_lorenz["z_init"]]), 
                params=np.array([input_lorenz["rho"], input_lorenz["beta"], input_lorenz["sigma"]]))
    data = np.concatenate((t[:, np.newaxis], ts), axis=1)
    print(data.shape)
    np.savetxt("Results/results_lorenz/"+filename, data, delimiter=' ', fmt='%f')
    

def plot_input_space(filenames, labels, savename, testkeys):
    fields = []
    for filename, value in zip(filenames, testkeys):
        field, _ = get_data_lorenz63(value, filename)
        fields.append(field)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, field in enumerate(fields):
        ax.scatter(field[:,0], field[:, 1], field[:,2], label=labels[i], alpha=0.5, s=2)
    fig.suptitle("Input space")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if len(labels) > 1:
        ax.legend()
    fig.savefig("images/Lorenz/"+savename+".pdf", dpi=200, bbox_inches='tight')
    fig.show()
    
def get_file_and_normalize_on_self(filename, normalize=True):
    array = np.loadtxt("Results/results_lorenz/"+filename, delimiter = ' ')
    fields = array[:, 1:]
    t = array[:, 0]
    if normalize:
        for i in range(3):
            fields[:, i] = (fields[:,i]-fields[:,i].mean())/fields[:,i].std()
    return fields, t 
    
    


if __name__=='__main__':
    filename = "dummy.dat"
    gen_and_save_data(filename)
    ts, t = get_data_lorenz63(test=True, name=filename)
    # filenames = [ "truth.dat", "test.dat", "valid.dat"]
    # labels = ["Training", "Test", "Validation"]
    # testkeys = [False, True, True]
    # plot_input_space(filenames, labels, "input_spaces", testkeys)
    # print(ts[25000,:])
    plot_data(t, ts, name="Validation")
    
    
    

    # dafi.run('lorenz.py', "EnKF", nsamples=2, ntime=None,
    #          save_level='time', save_dir='Results/results_dafi',
    #          verbosity=1, inputs_model={"input_file": "lorenz.in"})
    