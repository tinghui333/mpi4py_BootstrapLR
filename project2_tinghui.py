import time
import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.linear_model import LinearRegression

def main():
    time_0 = time.time()
    comm = MPI.COMM_WORLD

    ## Set up path and parameter
    # csv data file path
    data_path = '/gpfs/projects/AMS598/Projects2022/project2/project2_data.csv'

    # folder of current job
    curr_dir = '/gpfs/home/tinghwu/project2'
    # curr_dir = '/gpfs/projects/AMS598/class2022/tinghwu/project2'

    # number of bootstrap
    B = 10000

    b = int(B/comm.size)
    ci_range = int(np.ceil(B * 0.05 / 2))

    df = pd.read_csv(data_path)

    coef = np.empty((b, 5))
    time_1 = time.time()

    for i in range(b):
        ## Generate the sample dataset
        random_df = df.sample(n=len(df), replace=True)

        X = random_df.drop(['y'], axis=1)
        y = random_df['y']

        linear_regression = LinearRegression()
        model = linear_regression.fit(X, y)
        coef[i, :] = model.coef_

    comm.Barrier()
    time_2 = time.time()

    ## Gather the results from all nodes to root node
    # total_coef = comm.gather(coef)
    coef = np.sort(coef, axis=0)
    left_CI = comm.gather(coef[:ci_range, :])
    right_CI = comm.gather(coef[-ci_range:, :])

    if comm.rank == 0:
        left_CI = np.concatenate(left_CI, axis=0)
        left_CI = np.sort(left_CI, axis=0)
        right_CI = np.concatenate(right_CI, axis=0)
        right_CI = np.sort(right_CI, axis=0)

        # total_coef = np.concatenate(total_coef, axis=0)
        # np.save("coef.npy", total_coef)

        print("The left CI is {}".format(left_CI[ci_range, :]))
        print("The right CI is {}".format(right_CI[-ci_range, :]))

    time_3 = time.time()
    t_setup = time_1 - time_0
    t_regression = time_2 - time_1
    t_finish = time_3 - time_2
    print("Node {}: {}, {}, {}".format(comm.rank, t_setup, t_regression, t_finish))


if __name__ == '__main__':

    main()

    

