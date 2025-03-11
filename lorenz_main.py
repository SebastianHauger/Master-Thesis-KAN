import dafi


if __name__=='__main__':
    dafi.run('lorenz.py', "EnKF", nsamples=100, ntime=200, verbosity=1, inputs_model={"input_file": "lorenz.in"})
    