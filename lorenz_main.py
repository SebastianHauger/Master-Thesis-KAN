import dafi


if __name__=='__main__':
    dafi.run('lorenz.py', "EnKF", nsamples=2, ntime=None,
             save_level='time', save_dir='Results/results_dafi',
             verbosity=1, inputs_model={"input_file": "lorenz.in"})
    