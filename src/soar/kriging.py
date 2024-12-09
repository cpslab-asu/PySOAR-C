import numpy as np
import numpy.matlib as npm
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import eigh

np.set_printoptions(threshold=np.inf)


def OK_corr(corr_model, theta, D_X):
    d1 = D_X.shape[1]
    d2 = D_X.shape[2]
    d = theta.shape[0]
    # print(d1, d2,d)
    # print(theta)
    # print("*******************")
    # print(corr_model)
    if corr_model == 0:
        X = np.reshape(theta, (d,1, 1))
        # print(X)
        # print("*******************")
        rep_x = npm.tile(X, (1,d1, d2))
        # print(rep_x)
        # print("*******************")
        mul_part = (np.abs(D_X) * rep_x)
        # print(mul_part)
        # print("*******************")
        max_part = np.maximum(1-mul_part,0)
        # print(max_part)
        R_unshaped = np.prod(max_part,1)
        R = np.reshape(np.transpose(R_unshaped), (1,R_unshaped.shape[1],R_unshaped.shape[0]))
        # print("*******************")
        # print(R.shape)
        # print("*******************")

    elif corr_model == 1:
        X = np.reshape(theta,(d,1,1))
        # print(X)
        # print("*******************")
        rep_x = npm.tile(X,(1,d1, d2))
        # print(rep_x)
        # print("*******************")
        pre_mult = (-1*(np.abs(D_X)**corr_model))
        # print(pre_mult)
        # print("**********************")
        mul_part = (pre_mult * rep_x)
        # print(mul_part)
        # print("*******************")
        sum_part = np.sum(mul_part,0)
        R = np.exp(sum_part)
        # print("*******************")
        # print(R.shape)
        # print("*******************")

    elif corr_model ==  2:
        X = np.reshape(theta,(d,1,1))
        # print(X)
        # print("*******************")
        # rep_x = npm.tile(X,(1,d1, d2))
        # print(rep_x)
        # print("*******************")
        pre_mult = (-1*(np.abs(D_X)**corr_model))
        # print(pre_mult)
        # print("**********************")
        mul_part = (pre_mult * X)
        # print(mul_part)
        # print("*******************")
        sum_part = np.sum(mul_part,0)
        # print(sum_part)
        R = np.exp(sum_part)
        # print("*******************")
        # print(R)
        # print("*******************")

    elif corr_model == 3:
        X = np.reshape(theta,(d,1,1))
        
        # print(X)
        # print("*******************")
        rep_x = npm.tile(X,(1,d1, d2))
        
        term_1 = np.array((D_X<=(rep_x/2.)), dtype=bool)
        term_2_1 = 6*((D_X/rep_x)**2)
        term_2_2 = 6*((D_X/rep_x)**3)
        term_2 = 1 - term_2_1 + term_2_2
        term_3_1 = ((rep_x/2.)<D_X)
        term_3_2 = (D_X<=rep_x)
        term_3 = np.array((term_3_1 & term_3_2), dtype=bool)
        term_4 = (2*((1-D_X)/rep_x)**3)
        # print(term_1.shape)
        # print(term_2.shape)
        # print(term_3.shape)
        # print(term_4.shape)
        R = np.prod(term_1 * term_2 + term_3 * term_4,0)
        # print(term_4)
    #     R = prod(((D_X<=(T./2)).*(1-6*(D_X./T).^2+6*(D_X./T).^3)+((T./2)<D_X & D_X<=T).*(2*(1-D_X./T).^3)),3)

    return R


def OK_regr(data, regr_model):
    # Call the regression function for the MNEK model
    # X - design locations for the simulation inputs, size [k, d], k points with d
    # dimensions 
    # regr_model - the underlying regression model for the mean function:
    # regr_model = 0: constant mean function;
    # regr_model = 1: linear mean function;
    # regr_model = 2: quadratic mean function;

    num_samples, dim = data.shape
    
    if regr_model == 0:
        regr = np.ones((num_samples,1))
    elif regr_model == 1:
        regr = np.hstack((np.ones((num_samples,1)),data))
    elif regr_model == 2:
        mid = ((dim+1) * (dim+2))/2.0
        regr = np.hstack((np.ones((num_samples,1)), data, np.zeros((num_samples, (mid-dim-1)))))
        j = dim + 1
        q = dim
        for i in range(dim):
            regr[:,j+np.arange(0,q,1)] = np.multiply(np.repmat(X[:,i],1,q),X[:,i:n])
    return regr


def OK_Rlh_kd_nugget(params, num_samples, dim, D_X, Y, regr, corr_model, delta):
    # params = np.reshape(params, (dim,1))
    # if np.min(params[0:dim,0]) <= 0.001:
    #     f = math.inf
    #     return math.inf
    
    theta = np.reshape(params, (dim,1))
    # print(theta)
    R = OK_corr(corr_model, theta, D_X)
    R = R + delta * (np.eye(R.shape[0], R.shape[1]))
    CR = R
    U = (np.linalg.cholesky(CR)).transpose()
    
    L  = np.transpose(U)
    Linv = np.linalg.inv(L)
    Sinv = Linv.transpose() @ Linv
    

    beta = np.linalg.inv(np.transpose(regr) @ Sinv @ regr)@(np.transpose(regr) @ (Sinv @ Y))
    
    sigma_z = (1/num_samples) * (np.transpose(Y - (regr @ beta)) @ Sinv @ (Y - (regr@beta)))
    
    f = num_samples * (np.log(sigma_z)) + np.log(np.linalg.det(R))
    eps = 1e-15
    if np.isnan(f[0,0]) or np.isinf(f[0,0]):
        f = num_samples * (np.log(sigma_z+eps)) + np.log(np.linalg.det(R)+eps)
    
    return f[0,0]


def normalize_data(data):
    """normalizing data

    Args:
        data ([type]): 2D array with shape [num of samples X dimension]

    Returns:
        [normalized]: data noralized between 0 and 1 
    """
    min_data = np.min(data, 0)
    max_data = np.max(data, 0)

    normalized_data = (data - min_data)/((max_data-min_data)+1e-6)
    return normalized_data, min_data, max_data


def OK_Rmodel_kd_nugget(data_in, data_out, regr_model, corr_model, parameter_a):
    num_samples, dim = data_in.shape
    normal_data, min_data, max_data = normalize_data(data_in)

    tmp = 0
    D_x = np.zeros((dim,num_samples,num_samples))
    temp_d_x = np.zeros((num_samples*num_samples, dim))

    for h in range(dim):
        hh = 0
        for i in range(num_samples):
            for l in range(num_samples):
                D_x[h,i,l] = normal_data[i,h] - normal_data[l,h]
                temp_d_x[hh,h] = normal_data[i,h] - normal_data[l,h]
                hh = hh+1


    regr = OK_regr(normal_data,regr_model)
    beta_0 = np.linalg.lstsq(((np.transpose(regr) @ regr)), (np.transpose(regr) @ data_out), rcond=None)
    
    beta_0 = beta_0[0]
    sigma_z0 = np.var(data_out-(regr @ beta_0))
    theta_0 = np.zeros((dim,1))

    if corr_model == 0 or corr_model == 3:
        theta_0[:,0] = 0.5
    else:
        theta_0[:,0] = (np.log(2)/dim) * ((np.mean(np.abs(temp_d_x),0)+1e-10)**(-1*corr_model))

    corr = OK_corr(corr_model, theta_0, D_x)
    a = parameter_a
    cond_ = np.linalg.cond(corr, p=2)
    exp_ = np.exp(a)

    if np.allclose(corr, corr.T, rtol=1e-7, atol=1e-10):
        eigen_v = eigh(corr, eigvals_only=True, subset_by_index=[corr.shape[0]-1, corr.shape[0]-1])[0]

    if cond_ == np.inf:
        delta_lb = 0
    else:
        delta_lb = np.maximum(((eigen_v * (cond_ - exp_))/(cond_ * (exp_ - 1))),0)

    lob_sigma_z = 0.00001*sigma_z0
    lob_theta = 0.001*np.ones((dim,1)) 

    lower_bound_theta = np.ndarray.flatten(lob_theta)
    upper_bound_theta = np.full(lower_bound_theta.shape, np.inf)

    options = {'maxiter' : 1000000}
    lob = [lob_theta]
    bnds =  Bounds(lower_bound_theta, upper_bound_theta)
    fun = lambda p_in: OK_Rlh_kd_nugget(p_in, num_samples, dim, D_x, data_out, regr, corr_model, delta_lb)
    params = minimize(fun, np.ndarray.flatten(theta_0), method = 'Nelder-Mead', bounds = bnds, options = options)
    theta = np.reshape(params.x, (dim,1))
    R = OK_corr(corr_model, theta, D_x)
    CR = (R+delta_lb*np.eye(R.shape[0],R.shape[1]))
    U0 = np.linalg.cholesky(CR).transpose()
    CR=U0
    L  = np.transpose(U0)
    D_L = np.transpose(U0)
    Linv = np.linalg.inv(L)
    Rinv = Linv.transpose() @ Linv
    beta = np.linalg.inv(np.transpose(regr) @ Rinv @ regr)@(np.transpose(regr) @ (Rinv @ data_out))
    beta_v = np.linalg.inv(np.transpose(regr) @ Rinv @ regr)@(np.transpose(regr) @ Rinv)
    sigma_z = (1/num_samples) * (np.transpose(data_out - (regr @ beta)) @ Rinv @ (data_out - (regr@beta)))

    return {
        'sigma_z' :  sigma_z,
        'min_X' : min_data,
        'max_X' : max_data,
        'regr' :  regr,
        'beta' : beta,
        'beta_v' : beta_v,
        'theta' : theta,
        'X' : normal_data,
        'corr' : corr_model,
        'L' : L,
        'D_L' : D_L,
        'Z' : np.linalg.lstsq(L,(data_out-regr@beta), rcond=None),
        'Z_v' : np.linalg.lstsq(L,(np.eye(np.max(data_out.shape))-regr@beta_v), rcond=None),
        'Z_m' : np.linalg.inv(L),
        'DZ_m' : np.linalg.inv(D_L),
        'Rinv' : Rinv,
        'nugget' : delta_lb,
        'Y' : data_out
    }


def OK_Rpredict(gp_model, Xtest, regr_model):
    X = gp_model['X']
    min_X = gp_model['min_X']
    max_X = gp_model['max_X']
    num_samples, dim = X.shape
    theta = gp_model['theta']
    beta = gp_model['beta']
    Z = gp_model['Z']
    L = gp_model['L']
    Rinv = gp_model['Rinv']
    sigma_z = gp_model['sigma_z']
    corr_model = gp_model['corr']
    F = np.ones((num_samples,1))
    Ytrain = gp_model['Y']


    num_samples_test = Xtest.shape[0]

    regr_pred = OK_regr(Xtest,regr_model)

    normal_x_test = (Xtest - min_X)/((max_X-min_X)+1e-6)

    distXpred = np.zeros((dim,num_samples, num_samples_test))

    for h in range(dim):
        for i in range(num_samples):
            for j in range(num_samples_test):
                # distXpred[h,i,j] = normal_x_test[j,h] - X[i,h]
                distXpred[h,i,j] = X[i,h]- normal_x_test[j,h]

    R_pred = OK_corr(2, theta, distXpred)
    # print(R_pred)
    mse = np.full((num_samples_test,1),None)

    f = regr_pred*beta + R_pred.transpose()@(Rinv@(Ytrain-np.ones((num_samples,1))*beta))
    # print(f)
    FRFinv = 1/(F.transpose()@Rinv@F)
    
    Rinv_Rpred = Rinv@R_pred
    # print(FRFinv)
    for r in range(num_samples_test):
        OneMinusFcrossR = 1-(F.transpose()@Rinv_Rpred[:,r])
        
        mse_temp = sigma_z * (1 - R_pred[:,r].transpose()@Rinv_Rpred[:,r] + (OneMinusFcrossR).transpose()@FRFinv@(OneMinusFcrossR))
        mse[r,0] = np.sqrt(mse_temp[0,0])
        if np.isnan(mse[r,0]):
            raise Exception("Value of Krigin parameter too high, Try decreasing the value.")




    # for r in range(num_samples_test):
    #     # sigma_z * (1 - R_pred(:,r)'*Rinv*R_pred(:,r) + (1-F'*Rinv*R_pred(:,r))'*inv(F'*Rinv*F)*(1-F'*Rinv*R_pred(:,r)));
    #     term_1 = (1 - R_pred[:,r].transpose() @ Rinv @ R_pred[:,r])
    #     term_2 = (1-F.transpose() @ Rinv@R_pred[:,r]).transpose()
    #     term_3 = np.linalg.inv(F.transpose() @ Rinv @ F)
    #     term_4 = (1-F.transpose()@Rinv@R_pred[:,r])

    #     mse[r,0] = (sigma_z * (term_1 + term_2@term_3@term_4))[0,0]

    # mse = mse.flat
    return f, mse