import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.utils import shuffle
import matplotlib.backends.backend_pdf
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF #, RationalQuadratic, DotProduct, ExpSineSquared

def handle_ind(ind, n_cand_add, n_cand_del, X_org, Y_org, cvidx, est):
    n_not_selected = X_org.shape[1] - ind.shape[0]
    if n_not_selected < 20:
        coin = 2
    else:
        coin = np.random.choice(3, 1)
    # print(coin)
    # print(ind)
    if coin == 0:
        new_ind, op1, errs_stds = local_worst(ind, n_cand_del, X_org, Y_org, cvidx, est, ind)
        new_ind, op2, errs_stds = local_best(ind, n_cand_add, X_org, Y_org, cvidx, est, new_ind)
        op = op1 + ", " + op2
    elif coin == 1:
        new_ind, op, errs_stds = local_best(ind, n_cand_add, X_org, Y_org, cvidx, est, ind)
    else:
        new_ind, op, errs_stds = local_worst(ind, n_cand_del, X_org, Y_org, cvidx, est, ind)
    return new_ind, op, errs_stds

def local_extrema(ind, cand_inds, X_org, Y_org, cvidx, est, is_best):
    if is_best:
        testscore = np.inf
    else:
        testscore = -np.inf
    # print("cand_inds:", cand_inds)
    for cand_ind in cand_inds:
        if is_best:
            tempind = np.append(ind, cand_ind)
        else:
            tempind = np.delete(ind, cand_ind)
        # print("tempind: ", tempind)
        errs_stds = train_model(X_org[:, tempind], Y_org, cvidx, est)
        if cand_ind == cand_inds[0]:
            extrem_errs_stds = errs_stds
        cur_test_err, cur_test_std = errs_stds[2:]
        cur_score = cur_test_err + 0.1 * cur_test_std
        if testscore > cur_score and is_best:           # find local best
            testscore, extrem_ind, extrem_errs_stds = cur_score, cand_ind, errs_stds
        elif testscore < cur_score and not is_best:     # find local worst
            testscore, extrem_ind, extrem_errs_stds = cur_score, cand_ind, errs_stds
    return extrem_ind, extrem_errs_stds

def local_best(ind, n_cand_add, X_org, Y_org, cvidx, est, ind_temp):
    global cand_prob, cand_freq
    residue = np.delete(np.arange(X_org.shape[1]), ind)
    res_prob = np.take(cand_prob, residue)
    #res_prob = np.take(np.reciprocal(cand_freq), residue)
    #print("residue shape:", residue.shape[0], "n_cand_add", n_cand_add)
    if residue.shape[0] < n_cand_add:
        print("Residual features are not enough for candidates"
              "(size=" + str(n_cand_add)+"). Take all residue instead.")
        n_cand_add = int(residue.shape[0])
        print("New local settings: n_cand_add =", n_cand_add)
    cand_inds = np.random.choice(residue, n_cand_add, replace=False, p=res_prob/np.sum(res_prob))
    np.put(cand_prob, cand_inds, 0)
    cand_freq[cand_inds] += 1
    local_best_ind, errs_stds = local_extrema(ind, cand_inds, X_org, Y_org, cvidx, est, is_best=True)
    new_ind = np.append(ind_temp, local_best_ind)
    op = "Add Index " + str(local_best_ind)
    return new_ind, op, errs_stds

def local_worst(ind, n_cand_del, X_org, Y_org, cvidx, est, ind_temp):
    if ind.shape[0] < n_cand_del:
        print("Selected features are not enough for candidates"
              "(size=" + str(n_cand_del)+"). Take all selected instead.")
        n_cand_add = int(ind.shape[0])
        print("New local settings: n_cand_del =", n_cand_add)
    if ind.shape[0] <= 1:
        return ind
    cand_inds = np.random.choice(np.arange(ind.shape[0]), n_cand_del, replace=False)
    local_worst_ind, errs_stds = local_extrema(ind, cand_inds, X_org, Y_org, cvidx, est, is_best=False)
    local_worst_ind_temp = np.argwhere(ind_temp==ind[local_worst_ind])
    op = "Del Index " + str(ind_temp[local_worst_ind_temp])
    new_ind = np.delete(ind_temp, local_worst_ind_temp)
    return new_ind, op, errs_stds

def load_data(file_name, cols_skip = [], do_shuffle = False, show_features = False):
    data = pd.read_csv(file_name)     # read csv
    cols_keep = [col for col in data if col not in cols_skip]
    data = data[cols_keep]
    if show_features:
        names = data.columns.values
        print("Features:", names)
    # data = data.as_matrix(columns=data.columns[1:])     # get features and labels     #.as_matrix not supported for df
    data = data[data.columns[1:]].to_numpy()
    X_org = data[:, 1:]  # feature
    Y_org = data[:, 0]  # label
    if do_shuffle:
        X_org, Y_org = shuffle(X_org, Y_org)
    return X_org, Y_org

def ptb_cvidx(cvidx, n_exp, n_move, maxrate):
    ncv = len(cvidx) #cvidx is a list of [train, test]
    icvs = np.arange(ncv)
    if n_exp < ncv * n_move * maxrate:
        n_move = int(n_exp * maxrate * ncv)
    for i in range(ncv):
        icvs_residue = icvs[:i] + icvs[(i + 1):]
        test = cvidx[i][-1]
        idx_moves = np.random.choice(test, n_move, replace = False)
        #print(icvs_residue, n_move)
        icv_movetos = np.random.choice(icvs_residue, n_move)
        for idx_move, icv_moveto in zip(idx_moves, icv_movetos):
            np.append(cvidx[i][0], idx_move)
            np.delete(cvidx[i][1], np.where(cvidx[i][1] == idx_move))
            np.append(cvidx[icv_moveto][1], idx_move)
            np.delete(cvidx[icv_moveto][0], np.where(cvidx[icv_moveto][0] == idx_move))
    return cvidx

def train_model(cur_X, cur_Y, cvidx, est):
    trainerrs = []
    testerrs = []
    for train, test in cvidx:
        X, Y = cur_X[train], cur_Y[train]
        #print('Shape of X and Y', X.shape, Y.shape)
        est.fit(X, Y)
        Y_pred = est.predict(X)
        train_err = np.sqrt(metrics.mean_squared_error(Y, Y_pred))
        X_test, Y_test = cur_X[test], cur_Y[test]
        Y_test_pred = est.predict(X_test)
        test_err = np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred))
        trainerrs.append(train_err)
        testerrs.append(test_err)
    return np.mean(trainerrs), np.std(trainerrs), np.mean(testerrs), np.std(testerrs)

def cal_n_cand_add(algo, accm_iter, n_features, ind, n_cand_add):
    if algo == "org":
        n_cand_add = 1
    elif algo == "semi_rand_std5":
        n_cand_add = 5
    elif algo == "semi_rand_std10":
        n_cand_add = 10
    elif algo == "semi_rand_adp":
        n_cand_add = accm_iter + 1
    elif algo == "semi_rand_adp2":
        n_cand_add = (accm_iter + 1) * 2
    elif algo == "warm_start":
        if n_cand_add != 1:
            n_cand_add = n_features - (i + 1) * 3
            if n_cand_add < 4:
                n_cand_add = 1
    if n_cand_add > (n_features - ind.shape[0]) / 3:
        n_cand_add = int((n_features - ind.shape[0]) / 3)
    return n_cand_add

def cal_n_cand_del(i, n_features, ind, del_semi):
    n_selected = ind.shape[0]
    if del_semi:
        if n_selected < 8: #2 * (n_features - i):
            n_cand_del = 1 #int(n_selected / 3)
        else:
            n_cand_del = 4
    else:
        n_cand_del = 1
    return n_cand_del

def cal_n_iter(algo, max_n_itr_org, max_n_itr_warm_start, max_n_itr_new):
    if algo == "org":
        n_iter = max_n_itr_org
    elif algo == "warm_start":
        n_iter = max_n_itr_warm_start
    else:
        n_iter = max_n_itr_new
    return n_iter

def plot_data(metric_list, ylabel, lineStyle, color, alpha_eachrun, alpha_avg, algo, y_rmse_min, y_rmse_max):
    for metric in metric_list:
        plt.plot(range(len(metric)), metric, lineStyle, color=color, alpha=alpha_eachrun)
    metric_mean = np.mean(np.array(metric_list), axis=0)
    plt.plot(range(len(metric_mean)), metric_mean, lineStyle, color=color, alpha=alpha_avg,
             label=algo)
    plt.ylabel(ylabel)
    plt.xlabel('Number of Iterations')
    plt.legend(loc=1)
    if ylabel == "RMSE":
        plt.ylim(y_rmse_min, y_rmse_max)
    plt.grid(True)

def load_est(learner):
    if learner == "KRR":
        est = KernelRidge(alpha=1.0, kernel='rbf')
    elif learner == "GBDT":
        est = GradientBoostingRegressor(n_estimators = 50, learning_rate = 0.09, max_depth = 3)
    elif learner == "GP":
        noise_kernel = WhiteKernel()
        krq = RBF()
        est = GaussianProcessRegressor(kernel = krq + noise_kernel, normalize_y = True)
    else:
        print("Specified Learner \"" + learner + "\" Unknow. Use KRR instead.")
        est = KernelRidge(alpha=1.0, kernel='rbf')
    return est

def update_prob(cand_prob, ind, n_cand_add, n_features):
    cand_prob = cand_prob + n_cand_add / n_features * 50
    np.put(cand_prob, ind, 0)
    np.place(cand_prob, cand_prob > 1, 1)
    # print(np.sum(cand_prob))
    return cand_prob

# parameter settings
file_name = "data.csv"   # "dielectric.csv" file to process
enable_Elimination = False  # Conduct Randomized Feature Elimination instead of Selection
learner = "GP"              # KRR, GBDT
#algos = ["org", "semi_rand_std5", "semi_rand_std10", "semi_rand_adp", "semi_rand_adp2", "warm_start"]#, "warm_start", "warm_start_std5"]
algos = ["org", "semi_rand_std5", "semi_rand_adp", "warm_start"]#, "warm_start", "warm_start_std5"]
#algos = ["warm_start"]
n_run = 4                   # number of runs of feature selections
est = load_est(learner)     # load specific estimator
max_n_itr_new = 600         # number of iterations per new algorithm run (except original one and warm start)
max_n_itr_warm_start = 600  # maximum number of iterations of warm start
max_n_itr_org = 1000        # maximum number of iterations of org algorithm
n_samples_ini = 10          # size of subset of features to start with
n_fold = 5                  # number of folds for cross validation
colors = ["green", "red", "blue", "cyan", "magenta", "black", "yellow"]
cv_tb_sens = 2000           # perturb. sensitivity, start the perturb. after # of iter. when accuracy are not improved
cv_tb_maxrate = 0.3         # how many perturbation can happen in one time
alpha_eachrun = 0.2        # the transparency of each run in plot
alpha_avg = 0.5             # the transparency of the avg plot
y_rmse_max, y_rmse_min = 1.0, 0.40  # ylim of the RMSE plot

X_org, Y_org = load_data(file_name)
n_examples, n_features = X_org.shape
if enable_Elimination:
    n_samples_ini = n_features
print("Number of features:", n_features)
cand_prob = np.ones(n_features)
cand_freq = np.ones(n_features)

# Feature Selection
metrics_list = ["test_errs", "test_err_stds", "nidxs", "imprvs", "acc_unmvs", "runtimes_per_itr", "runtimes_acc"]
del_semis = [False]#, True]
lineStyles = ['-']#, '--']
for del_semi, lineStyle in zip(del_semis, lineStyles):
    colorind = 0
    plot_dict = dict()
    for algo in algos:     # run specific randomized feature selection algo
        n_iter = cal_n_iter(algo, max_n_itr_org, max_n_itr_warm_start, max_n_itr_new)
        for metric in metrics_list:
            plot_dict[metric] = []
        for run in range(n_run):
            n_cand_add = 0
            for metric in metrics_list:
                plot_dict[metric].append([])
            np.random.seed(run)
            cv = KFold(n_splits=n_fold, shuffle=True, random_state=run)  # cross validation setting
            cvidx = []
            for train, test in cv.split(X_org, Y_org):
                cvidx.append([train, test])
            if algo == "warm_start" and not enable_Elimination:
                ind = np.array([], dtype=np.int)
                ind, op, errs_stds = local_best(ind, n_features, X_org, Y_org, cvidx, est, ind)
                cur_train_err, cur_train_std, test_err, cur_test_std = errs_stds
            else:
                ind = np.random.choice(np.arange(n_features), n_samples_ini, replace=False)
                train_err, cur_train_std, test_err, cur_test_std = train_model(X_org[:, ind], Y_org, cvidx, est)
            accm_iter = 0
            ini_time = time.time()
            if not enable_Elimination:
                print("Initializing Features:", ind)
            print(algo, run, -1, None, ind.shape[0], train_err, test_err)
            for i in range(n_iter):
                start_time = time.time()
                n_cand_add = cal_n_cand_add(algo, accm_iter, n_features, ind, n_cand_add)
                n_cand_del = cal_n_cand_del(i, n_features, ind, del_semi)
                ind_new, op, errs_stds = handle_ind(ind, n_cand_add, n_cand_del, X_org, Y_org, cvidx, est)
                cur_train_err, cur_train_std, cur_test_err, cur_test_std = errs_stds
                # cur_train_err, cur_train_std, cur_test_err, cur_test_std = train_model(X_org[:, ind_new], Y_org, cvidx,
                #                                                                        est)
                accm_iter += 1
                if cur_test_err < test_err:
                    ind = ind_new
                    plot_dict["imprvs"][-1].append(test_err - cur_test_err)
                    train_err, test_err = cur_train_err, cur_test_err
                    accm_iter = 0
                    print("Improved:", op)
                    if not enable_Elimination:
                        print("New Feature Set:", ind)
                else:
                    plot_dict["imprvs"][-1].append(0)
                if accm_iter > cv_tb_sens:
                    cvidx = ptb_cvidx(cvidx, len(Y_org), accm_iter - cv_tb_sens, cv_tb_maxrate)
                    accm_iter = 0
                    print("Cross Validation Perturbation Conducted.")
                print(algo, run, i, ind_new.shape[0], ind.shape[0], train_err, test_err)
                cand_prob = update_prob(cand_prob, ind, n_cand_add, n_features)
                plot_dict["test_errs"][-1].append(test_err)
                plot_dict["test_err_stds"][-1].append(cur_test_std)
                plot_dict["nidxs"][-1].append(len(ind))
                plot_dict["acc_unmvs"][-1].append(accm_iter)
                plot_dict["runtimes_per_itr"][-1].append(time.time() - start_time)
                plot_dict["runtimes_acc"][-1].append(time.time() - ini_time)
            print(ind)

        least_test_errs = [item[-1] for item in plot_dict["test_errs"]]
        print("Semi Random Feature Elimination?", del_semi)
        print("Low test errs achieved:", least_test_errs)
        print("Average test error", np.mean(least_test_errs))
        print("Std of test errors", np.std(least_test_errs))

        metric_names = ["test_errs", "imprvs", "nidxs", "acc_unmvs", "runtimes_per_itr"]#, ["test_errs", "runtimes_acc"]]
        metric_ylabels = ["RMSE", "RMSE Improvement", "Number of Selected Features",
                         "Acc. No. of Iterations with No RMSE Improvements", "'Time Elapsed per Iteration (Sec)'"]
        figure_number = 0
        for metric_name, ylabel in zip(metric_names, metric_ylabels):
            plt.figure(figure_number)
            plot_data(plot_dict[metric_name], ylabel, lineStyle, colors[colorind], alpha_eachrun,
                      alpha_avg, algo, y_rmse_min, y_rmse_max)
            figure_number += 1

        plt.figure(figure_number)
        test_errs, runtimes_acc = plot_dict["test_errs"], plot_dict["runtimes_acc"]
        for test_err, runtime_acc in zip(test_errs, runtimes_acc):
            plt.plot(runtime_acc, test_err, lineStyle, color=colors[colorind], alpha=alpha_eachrun)
        test_err_mean = np.mean(np.array(test_errs), axis=0)
        runtime_acc_mean = np.mean(np.array(runtimes_acc), axis=0)
        plt.plot(runtime_acc_mean, test_err_mean, lineStyle, color=colors[colorind], alpha=alpha_avg, label=algo)
        plt.ylabel('RMSE')
        plt.xlabel('Time Elapsed (Sec)')
        plt.ylim(y_rmse_min, y_rmse_max)
        plt.legend(loc=1)
        plt.grid(True)

        colorind += 1

pdf = matplotlib.backends.backend_pdf.PdfPages("RandF_gp_bg_" + file_name[:-4] +"_may14_all.pdf")
for i in plt.get_fignums():
    fig = plt.figure(i)
    #fig.show()
    pdf.savefig(fig)
pdf.close()


