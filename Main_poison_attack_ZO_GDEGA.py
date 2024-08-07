from __future__ import print_function
import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16})
plt.switch_backend('agg')  ### running for CCC
import os
import h5py

#############################################################################
##################### Sub functions for data generation #####################
#############################################################################
def sigmoid_truncated(x):
    if 1.0/(1+np.exp(-x))>0.5:
        return 1
    else:
        return 0

def sigmoid_truncated_vec(x):
    binary_class = np.greater(1.0/(1+np.exp(-x)), 0.5)*1
    # if 1.0/(1+np.exp(-x))>0.5:
    #     return 1
    # else:
    #     return 0
    return binary_class

def generate_data(n,sigma2,x,noise_std,Data_path,filename):#generate a dataset
    m=len(x) ### dimension of coefficient with the ground truth x
    data=np.zeros((n,m+1)) ### n, # of training samples, m: feature vector dimension, m+1: label dimension
    sigma=np.sqrt(sigma2)
    for i in range(0,n):
        a=np.random.normal(0, sigma, m)
        noise=np.random.normal(0, noise_std)
        c=sigmoid_truncated((a.T).dot(x)+noise)
        data[i][0:m]=a
        data[i][m]=c
    np.savez(Data_path + "/" + filename, data=data)
    return data


def generate_train_and_test_data(train_ratio, poison_ratio, Data_path, filename, shuffle=False):#load generated data, get trainning and testing data
    data=np.load(Data_path + "/" + filename + ".npz")
    data=data['data']
    n=np.shape(data)[0]
    train_n=int(np.round(train_ratio*n))
    train_poison_n = int(np.round(poison_ratio*train_n))
    if shuffle:
        shuffled_indexes=list(range(0,n))
        random.shuffle(shuffled_indexes)
        data=data[shuffled_indexes]
    train_data=data[0:train_n]
    train_poison_data=train_data[0:train_poison_n]
    train_clean_data = train_data[train_poison_n:train_n]
    test_data=data[train_n:n]
    np.savez(Data_path + "/" + "train_4_2_SL_" +"PosRatio_" + str(poison_ratio) + "_" + filename + ".npz",train_data=train_data, train_poison_data = train_poison_data, train_clean_data = train_clean_data)
    np.savez(Data_path + "/" + "test_4_2_SL_" +"PosRatio_" + str(poison_ratio) + "_" + filename + ".npz",test_data=test_data)
    return train_data,train_poison_data,train_clean_data,test_data


def load_train_and_test_data(poison_ratio, Data_path, filename):#load training and testing data
    train_data=np.load(Data_path + "/" + "train_4_2_SL_" + "PosRatio_" + str(poison_ratio) + "_" + filename + ".npz")
    test_data=np.load(Data_path + "/" + "test_4_2_SL_" + "PosRatio_" + str(poison_ratio) + "_" + filename + ".npz")
    return train_data['train_data'],train_data['train_poison_data'],train_data['train_clean_data'], test_data['test_data']

def load_real_train_and_test_data(train_ratio, poison_ratio, Data_filename):
    data = h5py.File('D:/ZO-minimax-data/real_data/' + Data_filename + '.mat') 

    # data = h5py.File(data_path,"r")
    data = np.transpose(data['samples'])
    n = np.shape(data)[0]
    train_n = int(np.round(train_ratio*n))
    train_poison_n = int(np.round(poison_ratio*train_n))
    train_data=data[0:train_n,:]
    train_poison_data=train_data[0:train_poison_n,:]
    train_clean_data = train_data[train_poison_n:train_n,:]
    test_data=data[train_n:n,:]
    return train_data, train_poison_data, train_clean_data, test_data

def generate_index(length,b,iter,Data_path,filename): ### general
    index=[]
    for i in range(0,iter):
        temp=np.array(random.sample(range(0,length),b)) ### mini-batch scheme
        index.append(temp)  #### generate a list
    np.savez(Data_path + "/" +"batch_index_train_4_2_SL_" + filename + ".npz",index=index)
    return index

def load_index(Data_path,filename):
    index=np.load(Data_path + "/" + "batch_index_train_4_2_SL_" + filename + ".npz")
    index=index['index']
    return index

#############################################################################
##################### Sub functions for loss & projections #####################
#############################################################################

def loss_function_batch(delta,x,lambda_x, train_poison_data, train_clean_data, index_batch, index_poison):#compute loss for a batch
    length=np.shape(train_clean_data)[1]
    num_poison= np.shape(train_poison_data)[0]
    index=list(map(int,index_batch))
    index_poison=list(map(int,index_poison))
    a_clean=train_clean_data[index,0:length-1]
    a_poison = train_poison_data[index_poison, 0:length - 1] + np.matmul(np.ones((len(index_poison),1)),delta.reshape((1,-1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    # print(a_clean.shape, a_poison.shape)
    a = np.concatenate((a_clean,a_poison),axis=0)
    c_clean = train_clean_data[index,length-1]
    c_poison = train_poison_data[index_poison,length-1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))
    h=1.0/(1+np.exp(-a.dot(x)))
    value=( c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)) )/len(c)# - lambda_x*np.linalg.norm(x,2)**2 # general concave
    return value

def loss_function(delta,x,lambda_x, train_poison_data, train_clean_data):#compute loss for a dataset
    length=np.shape(train_clean_data)[1]
    num_poison= np.shape(train_poison_data)[0]
    a_clean=train_clean_data[:,0:length-1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison,1)),delta.reshape((1,-1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison),axis=0)

    c_clean =train_clean_data[:,length-1]
    c_poison = train_poison_data[:,length-1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h=1.0/(1+np.exp(-a.dot(x)))
    value=( c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)) )/len(c) #-lambda_x*np.linalg.norm(x,2)**2 # general concave
    return value

def project_inf(x,epsilon):
    x = np.greater(x,epsilon)*epsilon + np.less(x,-epsilon)*(-epsilon) \
    + np.multiply(np.multiply(np.greater_equal(x,-epsilon),np.less_equal(x,epsilon)),x)
    return x

#############################################################################
##################### Sub functions for  ZO algorithm ZO AG #####################
#############################################################################
def ZOPSGA(func,x0,step,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0 
    best_f=func(x0)
    sigma_dic =5
    flag=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter): 
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=x_opt+0.05*dx
        y_temp=func(x_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
    return x_opt

def ZOAGPy(func,x0,step,lr=0.02,iter=100,Q=10,t=1):
    D=len(x0)
    x_opt=x0 
    best_f=func(x0)
    sigma_dic =5
    flag=0
    # step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter): 
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=x_opt+lr*(dx - 0.1/(t+1)**0.25 * x_opt)

        y_temp=func(x_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
    return x_opt

def ZO_Semi_EGDA(func,x0,step,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0 
    best_f=func(x0)
    sigma_dic = 5
    flag=0
    # step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter): 
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=x_opt + lr*dx
        y_temp=func(x_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
    return x_opt

def ZO_Semi_EGDA2(func,yt,z2,step,lr=0.1,iter=100,Q=10): 
    D=len(yt)
    x_opt=yt 
    best_f=func(yt)
    sigma_dic = 5
    flag=0
    # step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter): 
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(z2+u*step)-func(z2))*u/step
            dx = dx + grad/Q
        x_temp=x_opt+lr*dx
        y_temp=func(x_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
    return x_opt

def ZOPSGD_bounded(func,x0,epsilon,step,lr=0.1,iter=100,Q=10,project=project_inf):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma_dic =5
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=project(x_opt-0.02*dx,epsilon)
        y_temp=func(x_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
    return x_opt

def ZOAGPx(func,x0,epsilon,step,lr=0.1,iter=100,Q=10,t=1,project=project_inf):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma_dic =5
    flag1=0
    # step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=project(x_opt-lr*100/(100 + t**0.5)*dx,epsilon)
        y_temp=func(x_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
    return x_opt

def ZO_Semi_EGDA_bounded(func,x0,epsilon,step,lr=0.1,iter=100,Q=10,project=project_inf):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma_dic =5
    flag1=0
    # step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=project(x_opt-lr*dx,epsilon)
        y_temp=func(x_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
    return x_opt



def AG_minmax_batch_Min_Max(func,delta0,x0,index,step,lr,q_rand_num,epsilon,iter,inner_iter=1):
    ### func： loss function
    ### delta0, x0: initial values
    ### index_batch: batch set for clean parts
    ### step: smooothing parameter
    ### lr: learning rate lr=[lr_delta,lr_x]
    ### iter: outer iterations
    ### inner_iter: inner iteration
    delta_opt=delta0
    x_opt=x0
    D_d=len(delta0)
    D_x=len(x0)
    flag=0
    best_f=func(delta0,x0,index[0])
    AG_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        #### record the initial point
        AG_iter_res[i][0:len(delta0)] = delta_opt  ### first D dimension is delta, then x
        AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        def func_xfixed(delta):
            return func(delta,x_opt,index[i])
        fdelta_pre = func_xfixed(delta_opt)
        delta_opt=ZOPSGD_bounded(func_xfixed,delta_opt,epsilon,step[0],lr[0],inner_iter,q_rand_num)
        temp_f=func_xfixed(delta_opt)
        if temp_f > fdelta_pre:
            print("Warning! Outer Min. Failed! ZO-Min-Max: Iter = %d, obj_pre = %3.4f, obj_post = %3.4f" % (
            i, fdelta_pre, temp_f))

        def func_deltafixed(x):
            return func(delta_opt,x,index[i])
        fx_pre = func_deltafixed(x_opt)
        x_opt=ZOPSGA(func_deltafixed,x_opt,step[1],lr[1],inner_iter,q_rand_num)
        fx_post = func_deltafixed(x_opt)
        if fx_post < fx_pre:
            print("Warning! Inner Max. Failed! ZO-Min-Max: Iter = %d, obj_pre = %3.4f, obj_post = %3.4f" % (i, fx_pre, fx_post) )

        # #### did not record the initial point
        # AG_iter_res[i][0:len(delta0)] = delta_opt  ### first D dimension is delta, then x
        # AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        if i%10 == 0:
            print("ZO-Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, q = %d, obj = %3.4f" % (i, lr[0], lr[1], q_rand_num, temp_f) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
            if  i > 9000:
                test = 1
        if temp_f<best_f:
            best_f=temp_f
        else:
            flag=flag+1
    return x_opt,AG_iter_res,AG_time

def AG_minmax_batch_AGP(func,delta0,x0,index,index_poison,step,lr,q_rand_num,epsilon,iter,inner_iter=1):
    ### func： loss function
    ### delta0, x0: initial values
    ### index_batch: batch set for clean parts
    ### step: smooothing parameter
    ### lr: learning rate lr=[lr_delta,lr_x]
    ### iter: outer iterations
    ### inner_iter: inner iteration
    delta_opt=delta0
    x_opt=x0
    D_d=len(delta0)
    D_x=len(x0)
    flag=0
    best_f=func(delta0,x0,index[0],index_poison[0])
    AG_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        #### record the initial point
        AG_iter_res[i][0:len(delta0)] = delta_opt  ### first D dimension is delta, then x
        AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        # update x
        def func_xfixed(delta):
            return func(delta,x_opt,index[i],index_poison[i])
        fdelta_pre = func_xfixed(delta_opt)
        delta_opt=ZOAGPx(func_xfixed,delta_opt,epsilon,step[0],lr[0],inner_iter,q_rand_num,i)
        temp_f=func_xfixed(delta_opt)
        if temp_f > fdelta_pre:
            print("Warning! Outer Min. Failed! ZO-AGP for Min-Max: Iter = %d, obj_pre = %3.4f, obj_post = %3.4f" % (
            i, fdelta_pre, temp_f))
        
        # update y
        def func_deltafixed(x):
            return func(delta_opt,x,index[i],index_poison[i])
        fx_pre = func_deltafixed(x_opt)
        x_opt=ZOAGPy(func_deltafixed,x_opt,step[1],lr[1],inner_iter,q_rand_num,i)
        fx_post = func_deltafixed(x_opt)
        if fx_post < fx_pre:
            print("Warning! Inner Max. Failed! ZO-AGP for Min-Max: Iter = %d, obj_pre = %3.4f, obj_post = %3.4f" % (i, fx_pre, fx_post) )

        #### did not record the initial point
        # AG_iter_res[i][0:len(delta0)] = delta_opt  ### first D dimension is delta, then x
        # AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        if i%10 == 0:
            print("ZO-AGP for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, q = %d, obj = %3.4f" % (i, lr[0], lr[1], q_rand_num, temp_f) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
            if  i > 9000:
                test = 1
        if temp_f<best_f:
            best_f=temp_f
        else:
            flag=flag+1
            #if flag%3==0:
            #    lr[0]=lr[0]*0.98
    return x_opt,AG_iter_res,AG_time

def AG_minmax_batch_EG(func,delta0,x0,index,index_poison,step,lr,q_rand_num,epsilon,iter,inner_iter=1):
    ### func： loss function
    ### delta0, x0: initial values
    ### index_batch: batch set for clean parts
    ### step: smooothing parameter
    ### lr: learning rate lr=[lr_delta,lr_x]
    ### iter: outer iterations
    ### inner_iter: inner iteration
    delta_opt=delta0
    x_opt=x0
    z_opt = x_opt 
    D_d=len(delta0)
    D_x=len(x0)
    flag=0
    best_f=func(delta0,x0,index[0],index_poison[0])
    AG_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter): # 50000
        AG_time[i]=time.time()
        #### record the initial point
        AG_iter_res[i][0:len(delta0)] = delta_opt  ### first D dimension is delta, then x
        AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        ########################## ZO Gradient Descnet Extragradient Ascent Algorithm #######################
        x_t_1 = x_opt # y_{t}
        delta_t_1 = delta_opt # x_{t}
        z_t_1 = z_opt # z_{t}
        # update x
        def func_xfixed(delta):
            return func(delta,z_t_1,index[i],index_poison[i])
        fdelta_pre = func_xfixed(delta_t_1)
        delta_t_2 = ZO_Semi_EGDA_bounded(func_xfixed,delta_t_1,epsilon,step[0],lr[0],inner_iter,q_rand_num) # x_{t+1}
        temp_f=func_xfixed(delta_t_2)

        # update z
        def func_deltafixed(x):
            return func(delta_t_1,x,index[i],index_poison[i])
        fx_pre = func_deltafixed(x_t_1)
        z_t_2 = ZO_Semi_EGDA(func_deltafixed,x_t_1,step[1],lr[1],inner_iter,q_rand_num) # z_{t+1}
        fx_post = func_deltafixed(z_t_2)

        # update y 
        def func_deltafixed(x):
            return func(delta_t_2,x,index[i],index_poison[i])
        fx_pre = func_deltafixed(z_t_2)
        x_t_2 = ZO_Semi_EGDA2(func_deltafixed,x_t_1,z_t_2,step[1],lr[1],inner_iter,q_rand_num) # y_{t+1}
        fx_post = func_deltafixed(x_t_2)
        
        x_opt = x_t_2
        z_opt = z_t_2
        delta_opt = delta_t_2

        #### did not record the initial point
        # AG_iter_res[i][0:len(delta0)] = delta_opt  ### first D dimension is delta, then x
        # AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        if i%10 == 0:
            print("ZO-semi-EG for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, q = %d, obj = %3.4f" % (i, lr[0], lr[1], q_rand_num, temp_f) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
            if  i > 9000:
                test = 1
        if temp_f<best_f:
            best_f=temp_f
        else:
            flag=flag+1
            #if flag%3==0:
            #    lr[0]=lr[0]*0.98
    return x_opt,AG_iter_res,AG_time

def AG_run_batch_Min_Max(func,delta0,x0,index,epsilon,step,lr,q_rand, iter,inner_iter=1):
    x_opt,AG_iter_res,AG_time=AG_minmax_batch_Min_Max(func,delta0,x0,index,step,lr,q_rand, epsilon,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time

def AG_run_batch_AGP(func,delta0,x0,index,index_poison,epsilon,step,lr,q_rand, iter,inner_iter=1):
    x_opt,AG_iter_res,AG_time=AG_minmax_batch_AGP(func,delta0,x0,index,index_poison,step,lr,q_rand, epsilon,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time

def AG_run_batch_EG(func,delta0,x0,index,index_poison,epsilon,step,lr,q_rand, iter,inner_iter=1):
    x_opt,AG_iter_res,AG_time=AG_minmax_batch_EG(func,delta0,x0,index,index_poison,step,lr,q_rand, epsilon,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time

def AG_main_batch_SL(train_poison_data, train_clean_data,  x_ini, delta_ini,  eps_perturb, n_iters, x_gt, index_batch, lambda_x, lr_delta, lr_x, q_rand_num, Data_path, filename=None ):
    n_x = len(x_ini)
    lr_x = np.min([lr_x, 1/np.sqrt(n_x)])
    lr_delta = np.min([lr_delta, 1 / np.sqrt(n_x)])
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_AG(delta,x,index):
        loss=loss_function_batch(delta,x,lambda_x, train_poison_data, train_clean_data, index)
        return loss

    print("##################################################################")
    print("ZO-Min-Max method")
    time_start=time.time()

    ### highlight: lr=[lr_delta,lr_x]
    x_opt,AG_iter_res,AG_time=AG_run_batch_Min_Max(loss_AG,delta0,x0,index_batch,eps_perturb,step=[0.001,0.001],lr=[lr_delta,lr_x],q_rand = q_rand_num , iter=n_iters,inner_iter=1)

    time_end=time.time()
    print('Time cost of ZO-Min-Max:',time_end-time_start,"s")

    if filename==None:
        np.savez(Data_path + "/" +"ZO_Min_Max_4_2_SL.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    else:
        np.savez(Data_path + "/" + "ZO_Min_Max_4_2_SL_"+filename+".npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return x_opt, AG_iter_res, AG_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]

def AG_main_batch_SL_AGP(train_poison_data, train_clean_data,  x_ini, delta_ini,  eps_perturb, step, n_iters, x_gt, index_batch, index_poison, lambda_x, lr_delta, lr_x, q_rand_num, Result_path, filename=None ):
    n_x = len(x_ini)
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_AG(delta,x,index,index_poison):
        loss=loss_function_batch(delta,x,lambda_x, train_poison_data, train_clean_data, index, index_poison)
        return loss

    print("##################################################################")
    print("ZO-AGP method")
    time_start=time.time()

    ### highlight: lr=[lr_delta,lr_x]
    x_opt,AG_iter_res,AG_time=AG_run_batch_AGP(loss_AG,delta0,x0,index_batch, index_poison, eps_perturb,step,lr=[lr_delta,lr_x],q_rand = q_rand_num , iter=n_iters, inner_iter=1)

    time_end=time.time()
    print('Time cost of ZO-AGP:',time_end-time_start,"s")

    if filename==None:
        np.savez(Result_path + "/" +"ZO_AGP_results.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    else:
        # filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j)+"_AGP"
        np.savez(Result_path + "/" + "ZO_AGP_results_"+filename+".npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return x_opt, AG_iter_res, AG_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]

def AG_main_batch_SL_EG(train_poison_data, train_clean_data, x_ini, delta_ini, eps_perturb, step, n_iters, x_gt, index_batch, index_poison, lambda_x, lr_delta, lr_x, q_rand_num, Result_path, filename=None ):
    n_x = len(x_ini)
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_AG(delta,x,index,index_poison):
        loss=loss_function_batch(delta,x,lambda_x, train_poison_data, train_clean_data, index, index_poison)
        return loss

    print("##################################################################")
    print("ZO-GDEGA method")
    time_start=time.time()

    ### highlight: lr=[lr_delta,lr_x]
    x_opt,AG_iter_res,AG_time=AG_run_batch_EG(loss_AG,delta0,x0,index_batch,index_poison, eps_perturb,step,lr=[lr_delta,lr_x],q_rand = q_rand_num , iter=n_iters,inner_iter=1)

    time_end=time.time()
    print('Time cost of ZO-GDEGA:',time_end-time_start,"s")

    if filename==None:
        np.savez(Result_path + "/" +"ZO_GDEGA_results.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    else:
        # filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j)+"_semi_EG"
        np.savez(Result_path + "/" + "ZO_GDEGA_results_" + filename+".npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return x_opt, AG_iter_res, AG_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]


#############################################################################
##################### Sub functions for  First Order algorithm FO AG #####################
#############################################################################

################### FO gradient computation #####################
def loss_derivative_x_index(delta,x,lambda_x,train_poison_data, train_clean_data,index,index_poison):
    length=np.shape(train_clean_data)[1]
    num_poison= np.shape(train_poison_data)[0]
    index=list(map(int,index))
    index_poison=list(map(int,index_poison))
    a_clean=train_clean_data[index,0:length-1]
    a_poison = train_poison_data[index_poison, 0:length - 1] + np.matmul(np.ones((len(index_poison),1)),delta.reshape((1,-1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison))

    c_clean =train_clean_data[index,length-1]
    c_poison = train_poison_data[index_poison,length-1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h_poison = 1.0 / (1 + np.exp(-a_poison.dot(x)))
    h=1.0/(1+np.exp(-a.dot(x)))

    derivative_x = -(((h-c).T).dot(a)).T/len(c) #-2*lambda_x*x # general concave
    # derivative_delta = -np.sum(h_poison-c_poison)/len(c_poison)*x
    # derivative = np.concatenate(derivative_delta.reshape(-1),derivative_x.reshape(-1))
    #print(derivative)
    return derivative_x

def loss_derivative_x(delta,x,lambda_x,train_poison_data, train_clean_data):
    length=np.shape(train_clean_data)[1]
    num_poison= np.shape(train_poison_data)[0]
    a_clean=train_clean_data[:,0:length-1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison,1)),delta.reshape((1,-1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison))
    c_clean =train_clean_data[:,length-1]
    c_poison = train_poison_data[:,length-1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h_poison = 1.0 / (1 + np.exp(-a_poison.dot(x)))
    h=1.0/(1+np.exp(-a.dot(x)))

    derivative_x = -(((h-c).T).dot(a)).T/len(c) #-2*lambda_x*x # general concave
    # derivative_delta = -np.sum(h_poison-c_poison)/len(c_poison)*x
    # derivative = np.concatenate(derivative_delta.reshape(-1),derivative_x.reshape(-1))
    #print(derivative)
    return derivative_x

def loss_derivative_delta_index(delta,x,lambda_x,train_poison_data, train_clean_data,index,index_poison):
    length = np.shape(train_clean_data)[1]
    num_poison = np.shape(train_poison_data)[0]
    index = list(map(int, index))
    index_poison = list(map(int, index_poison))
    a_clean = train_clean_data[index, 0:length - 1]
    a_poison = train_poison_data[index_poison, 0:length - 1] + np.matmul(np.ones((len(index_poison), 1)), delta.reshape((1, -1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison))

    c_clean = train_clean_data[index, length - 1]
    c_poison = train_poison_data[index_poison, length - 1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h_poison = 1.0 / (1 + np.exp(-a_poison.dot(x)))
    h = 1.0 / (1 + np.exp(-a.dot(x)))

    # derivative_x = -(((h - c).T).dot(a)).T / len(c) - 2 * lambda_x * x
    if len(c_poison) == 0:
        derivative_delta = 0*x
    else:
        derivative_delta = -np.sum(h_poison-c_poison)/len(c_poison)*x
    # derivative = np.concatenate(derivative_delta.reshape(-1),derivative_x.reshape(-1))
    return derivative_delta

def loss_derivative_delta(delta,x,lambda_x,train_poison_data, train_clean_data):
    length = np.shape(train_clean_data)[1]
    num_poison = np.shape(train_poison_data)[0]
    a_clean = train_clean_data[:, 0:length - 1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison, 1)), delta.reshape((1, -1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison))

    c_clean = train_clean_data[:, length - 1]
    c_poison = train_poison_data[:, length - 1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h_poison = 1.0 / (1 + np.exp(-a_poison.dot(x)))
    h = 1.0 / (1 + np.exp(-a.dot(x)))

    # derivative_x = -(((h - c).T).dot(a)).T / len(c) - 2 * lambda_x * x
    derivative_delta = -np.sum(h_poison-c_poison)/len(c_poison)*x
    # derivative = np.concatenate(derivative_delta.reshape(-1),derivative_x.reshape(-1))
    return derivative_delta


################### FO optimizer
def FO_run_batch(func,train_poison_data, train_clean_data,delta0,x0,index,index_poison,epsilon,lambda_x,lr,iter=100,project=project_inf):
    lr=np.array(lr)     # lr: learning rate lr=[lr_delta,lr_x]
    FO_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    FO_time=np.zeros(iter)
    D_d=len(delta0)
    D_x=len(x0)
    delta_opt=delta0
    x_opt=x0
    flag1=0
    best_f = func(delta_opt, x_opt, index[0], index_poison[0])
    for i in range(0,iter):

        FO_time[i]=time.time()
        FO_iter_res[i][0:D_d]=delta_opt
        FO_iter_res[i][D_d:D_d+D_x]=x_opt

        ddelta=loss_derivative_delta_index(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data,index[i],index_poison[i])
        delta_opt=project(delta_opt-lr[0]*100/(100 + i**0.5) * ddelta,epsilon)


        dx=loss_derivative_x_index(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data,index[i],index_poison[i])
        x_opt=x_opt+0.05*(dx - lr[1]/(i+1)**0.25 * x_opt)
        # y_temp=func(delta_opt,x_opt,index[i])
        # if y_temp>func(delta_opt,x_opt,index[i]):
        #     x_opt=x_temp



        y_temp=func(delta_opt,x_opt,index[i],index_poison[i])
        if i%10 == 0:
            print("FO-AGP for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
        if y_temp<best_f:
            best_f=y_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt,FO_iter_res,FO_time

#### FO cal
def FO_main_batch_SL(train_poison_data, train_clean_data, x_ini,delta_ini, eps_perturb,
                     n_iters, x_gt, index_batch, index_poison, lambda_x, lr_delta, lr_x, Result_path,  filename=None ):
    n_x = len(x_ini)
    # lr_x = np.min([lr_x, 1/np.sqrt(n_x)])
    # lr_delta = np.min([lr_delta, 1 / np.sqrt(n_x)])
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_FO(delta,x,index,index_poison):
        loss=loss_function_batch(delta,x,lambda_x, train_poison_data, train_clean_data, index, index_poison)
        return loss

    print("##################################################################")
    print("FO-AG method")
    time_start=time.time()

    x_opt,FO_iter_res,FO_time=FO_run_batch(loss_FO,train_poison_data, train_clean_data,delta0,x0,index_batch, index_poison,
                                           eps_perturb,lambda_x,lr=[lr_delta,lr_x],iter=n_iters,project=project_inf)

    time_end=time.time()
    print('Time cost of FO-AG:',time_end-time_start,"s")

    if filename==None:
        np.savez(Result_path + "/" +"FOAG_4_2_SL.npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    else:
        # filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j)+"_AGP"
        np.savez(Result_path + "/" +"FOAG_4_2_SL_"+filename+".npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    return x_opt, FO_iter_res, FO_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]



#############################################################################
#################### Sub functions for Non-Adv FO approach #############################
#############################################################################
def Soft_thresholding(a, theta):
    for i in range(0, len(a)):
        if a[i] > theta:
            a[i] = a[i] - theta
        elif a[i] >= -theta:
            a[i] = 0
        else:
            a[i] = a[i] + theta
    return a
#### non-min-max first order
def FO_nonAdv_run_batch(func,train_poison_data, train_clean_data,x0,index,index_poison,lambda_x,lr,iter=100):
    lr=np.array(lr)
    BL_iter_res=np.zeros((iter,2*len(x0)))
    BL_time=np.zeros(iter)
    D_x=len(x0)
    D_d=D_x
    delta_opt=np.zeros(D_d)
    x_opt=x0
    best_f=func(x_opt,index[0],index_poison[0])
    flag1=0

    for i in range(0,iter):
        BL_time[i]=time.time()
        BL_iter_res[i][0:D_d]=delta_opt
        BL_iter_res[i][D_d:D_d+D_x]=x_opt

        dx=loss_derivative_x_index(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data,index[i],index_poison[i])

        x_opt=x_opt+dx*lr[1]
        y_temp=func(x_opt,index[i],index_poison[i])
        if y_temp>best_f:
            best_f=y_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
        if i%10 == 0:
            print("Non-Adv FO: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
    return x_opt,BL_iter_res,BL_time


### run
def FO_nonAdv_main_batch_SL(train_poison_data, train_clean_data, x_ini, delta_ini, eps_perturb,
                     n_iters, x_gt, index_batch, index_poison, lambda_x,  lr_delta, lr_x, Result_path, filename=None):
    n_x = len(x_ini)
    # lr_x = np.min([lr_x, 1 / np.sqrt(n_x)])
    # lr_delta = np.min([lr_delta, 1 / np.sqrt(n_x)])
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_FO(x, index, index_poison):
        loss = loss_function_batch(np.zeros(len(delta0)), x, lambda_x, train_poison_data, train_clean_data, index, index_poison)
        return loss


    print("##################################################################")
    print("FO-nonAdv method")
    time_start = time.time()

    x_opt, FO_nonAdv_iter_res, FO_nonAdv_time = FO_nonAdv_run_batch(loss_FO, train_poison_data, train_clean_data,
                                                                    x0, index_batch, index_poison,
                                                                     lambda_x, lr=[lr_delta, lr_x], iter=n_iters)

    time_end = time.time()
    print('Time cost of FO-nonAdv:', time_end - time_start, "s")

    if filename == None:
        np.savez(Result_path + "/" +"FOnonAdv_4_2_SL.npz", x_gt=x_gt, FO_nonAdv_res=FO_nonAdv_iter_res, FO_nonAdv_time=FO_nonAdv_time)
    else:
        # filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j)
        np.savez(Result_path + "/" +"FOnonAdv_4_2_SL_" + filename + ".npz", x_gt=x_gt, FO_nonAdv_iter_res=FO_nonAdv_iter_res, FO_nonAdv_time=FO_nonAdv_time)
    return x_opt, FO_nonAdv_iter_res, FO_nonAdv_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]


### retrain after obtaining poisoned data using FO solver
def FO_retrain_poison(train_poison_data, train_clean_data, x_ini, delta_opt,
                     n_iters, lambda_x,  lr_x,  Retrain_epoch = 1):

    n_x = len(x_ini)
    # lr_x = np.min([lr_x, 1 / np.sqrt(n_x)])
    x_opt = x_ini.copy()
    num_train = np.shape(train_clean_data)[0]
    num_train_poison = np.shape(train_poison_data)[0]

    def loss_retrain(x, index_clean = np.arange(0, num_train).astype(int), index_poison = np.arange(0, num_train_poison).astype(int)):
        loss = loss_function_batch(delta_opt, x, lambda_x, train_poison_data, train_clean_data, index_clean, index_poison)
        return loss

    # best_f=loss_retrain(x_opt)

    retrain_iter_res=np.zeros((n_iters,n_x))
    batch_size = 1000 
    #### update x
    for i in range(0, n_iters):
        index_clean=np.array(random.sample(range(0,np.size(train_clean_data,0)),100)) ### mini-batch scheme
        index_poison=np.array(random.sample(range(0,np.size(train_poison_data,0)),100)) ### mini-batch scheme
        dx=loss_derivative_x_index(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data, index_clean, index_poison)
        x_opt = x_opt+dx*lr_x

        retrain_iter_res[i][:] = x_opt

        y_temp=loss_retrain(x_opt, index_clean, index_poison)

        # if y_temp>best_f:
        #     best_f=y_temp
        if i%100 == 0:
            print("Retrain FO: Retrain_epoch = %d, Iter = %d, lr_x=%f, obj = %3.4f" % (Retrain_epoch, i, lr_x, y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))

        # np.savez("Retrain_4_2_SL_" + filename + ".npz", retrain_iter_res = retrain_iter_res)
    return x_opt ### x_opt, AG_iter_res = [delta_iter, x_iter]


########################################################################################
######################## Subfunctions for Experiments Plot #############################
########################################################################################

########### loss for all data and all obtained variables
def all_loss(iter_res,lambda_x,train_poison_data, train_clean_data,test_data):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_poison_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    all_train_loss=np.zeros(iter)
    # all_test_loss=np.zeros(iter)
    for i in range(0,iter):
        all_train_loss[i]=loss_function(iter_res_delta[i],iter_res_x[i],lambda_x,train_poison_data, train_clean_data)
    return all_train_loss



def acc_for_D(x,data):#compute loss for a dataset
    length=np.shape(data)[1]
    a=data[:,0:length-1]
    c=data[:,length-1]
    # acc=0
    # for i in range(0,np.shape(data)[0]):
    #     if abs(c[i]-sigmoid_truncated((a[i].T).dot(x)))<1e-2:
    #         acc=acc+1
    # acc=acc/np.shape(data)[0]
    predict_results = sigmoid_truncated_vec(a.dot(x))
    acc = 1-np.sum(np.abs(c - predict_results))/np.shape(data)[0]
    return acc

def acc_for_PoisonD(x, delta, poison_data, clean_data):  # compute loss for a dataset
    length = np.shape(clean_data)[1]
    num_poison= np.shape(poison_data)[0]
    a_clean = clean_data[:, 0:length - 1]
    a_poison = poison_data[:, 0:length - 1] + np.matmul(np  .ones((num_poison, 1)), delta.reshape((1, -1))) 
    a = np.concatenate((a_clean, a_poison), axis=0)
    c_clean = clean_data[:, length - 1]
    c_poison = poison_data[:, length - 1]
    c = np.concatenate((c_clean, c_poison))
    # acc = 0
    # for i in range(0, np.shape(a)[0]):
    #     if abs(c[i] - sigmoid_truncated((a[i].T).dot(x))) < 1e-2:
    #         acc = acc + 1
    # acc = acc / np.shape(a)[0]
    predict_results = sigmoid_truncated_vec(a.dot(x))
    acc = 1-np.sum(np.abs(c - predict_results))/np.shape(a)[0]
    return acc

def all_acc(iter_res,train_poison_data, train_clean_data,test_data):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    all_train_acc=np.zeros(iter)
    all_test_acc=np.zeros(iter)
    for i in range(0,iter):
        all_test_acc[i]=acc_for_D(iter_res_x[i],test_data)
        ### poisoned training data
        all_train_acc[i] = acc_for_PoisonD(iter_res_x[i], iter_res_delta[i],train_poison_data, train_clean_data)
    return all_train_acc, all_test_acc

def all_acc_retrain(iter_res,train_poison_data, train_clean_data,test_data, lambda_x):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    all_train_acc=np.zeros(iter)
    all_test_acc=np.zeros(iter)
    iters_retrain = 1000
    for i in range(0,iter): 
        ### update iter_res_x[i] based on iter_res_delta[i]
        xi_retrain = FO_retrain_poison(train_poison_data, train_clean_data, iter_res_x[i], iter_res_delta[i],
                                             iters_retrain, lambda_x, 0.1, i)
        all_test_acc[i] = acc_for_D(xi_retrain, test_data) 
        ### poisoned training data
        all_train_acc[i] = acc_for_PoisonD(xi_retrain, iter_res_delta[i], train_poison_data, train_clean_data)
    return all_train_acc, all_test_acc


def stationary_condition(iter_res,train_poison_data, train_clean_data,lambda_x,alpha,beta,epsilon):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    G=np.zeros((iter,2*D))
    for i in range(0,iter):
        delta_opt=iter_res_delta[i]
        x_opt=iter_res_x[i]
        G[i][0:D]=-loss_derivative_x(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data) 
        # proj_alpha=project_inf(delta_opt-alpha*loss_derivative_delta(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data),epsilon=epsilon)
        # G[i][D:2*D]=1/alpha*(delta_opt-proj_alpha)
        proj_alpha = project_inf(delta_opt - loss_derivative_delta(delta_opt, x_opt, lambda_x, train_poison_data, train_clean_data),epsilon=epsilon)
        G[i][D:2 * D] = 1 / 1 * (delta_opt - proj_alpha) 
    return np.linalg.norm(G,ord=2,axis=1)

def mean_std(data):
    n=len(data)
    iter=len(data[0])
    mean=np.zeros(iter)
    std=np.zeros(iter)
    for i in range(0,iter):
        iter_data=np.zeros(n)
        for j in range(0,n):
            iter_data[j]=np.array(data[j][i])
        mean[i]=np.mean(iter_data)
        std[i]=np.std(iter_data)
    return mean,std

def plot_shaded_logx(x_plot, mean,std,ax_handle, color):
    # iter=len(mean)
    mean=np.array(mean)
    std=np.array(std)
    low=mean-std
    high=mean+std
    p1, = ax_handle.semilogx(x_plot,mean, linewidth=2, color = color)
    ax_handle.fill_between(x_plot,low,high,alpha=0.3, color = color)
    return p1

def plot_shaded(x_plot, mean,std,ax_handle, color):
    # iter=len(mean)
    mean=np.array(mean)
    std=np.array(std)
    low=mean-std
    high=mean+std
    p1, = ax_handle.plot(x_plot,mean, linewidth=2, color = color)
    ax_handle.fill_between(x_plot,low,high,alpha=0.3, color = color)
    return p1

def plot_threeline_shaded_logx(x_plot,data1_mean,data1_std,data2_mean,data2_std,data3_mean,data3_std,xlabel,ylabel,legend=["ZO-Min-Max","FO-Min-Max","FO-NoPoison"],loc='upper left',filename=None):
    # plt.figure()
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded_logx(x_plot,data1_mean,data1_std,ax_handle)
    p2=plot_shaded_logx(x_plot,data2_mean,data2_std,ax_handle)
    p3=plot_shaded_logx(x_plot,data3_mean,data3_std,ax_handle)
    ax_handle.legend([p1, p2, p3], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)
        # plt.savefig(filename)
    # plt.show()
    # plt.close()

def plot_fourline_shaded_logx(x_plot,data1_mean,data1_std,color1,data2_mean,data2_std,color2,data3_mean,data3_std,color3,data4_mean,data4_std,color4,xlabel,ylabel,legend=["ZO-AGP","ZO-Semi-EGDA","FO-Min-Max","FO-NoPoison"],loc='upper left',fig_dir = 'Results_figures', filename=None):
    # plt.figure()
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded_logx(x_plot,data1_mean,data1_std,ax_handle,color1)
    p2=plot_shaded_logx(x_plot,data2_mean,data2_std,ax_handle,color2)
    p3=plot_shaded_logx(x_plot,data3_mean,data3_std,ax_handle,color3)
    p4=plot_shaded_logx(x_plot,data4_mean,data4_std,ax_handle,color4)
    ax_handle.legend([p1, p2, p3, p4], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join(fig_dir,filename)
        plt.savefig(my_path)
        # plt.savefig(filename)
    # plt.show()
    # plt.close()
def plot_fiveline_shaded_logx(x_plot,data1_mean,data1_std,color1,data2_mean,data2_std,color2,data3_mean,data3_std,color3,data4_mean,data4_std,color4,data5_mean,data5_std,color5,xlabel,ylabel,legend=["ZO-Min-Max","ZO-AGP","ZO-Semi-EGDA","FO-Min-Max","FO-NoPoison"],loc='upper left',fig_dir = 'Results_figures', filename=None):
    # plt.figure()
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded_logx(x_plot,data1_mean,data1_std,ax_handle,color1)
    p2=plot_shaded_logx(x_plot,data2_mean,data2_std,ax_handle,color2)
    p3=plot_shaded_logx(x_plot,data3_mean,data3_std,ax_handle,color3)
    p4=plot_shaded_logx(x_plot,data4_mean,data4_std,ax_handle,color4)
    p5=plot_shaded_logx(x_plot,data5_mean,data5_std,ax_handle,color5)
    ax_handle.legend([p1, p2, p3, p4, p5], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join(fig_dir,filename)
        plt.savefig(my_path)
        # plt.savefig(filename)
    # plt.show()
    # plt.close()

def plot_sixline_shaded_logx(x_plot,data1_mean,data1_std,data2_mean,data2_std,data3_mean,data3_std,data4_mean,data4_std,data5_mean,data5_std,data6_mean,data6_std,xlabel,ylabel,legend=["ZO-AGP: q=5","ZO-semi-EGDA: q=5","ZO-AGP: q=20","ZO-semi-EGDA: q=20","FO-AGP","FO-No Poison"],loc='upper left',fig_dir='Results_figures',filename=None):
    # plt.figure()
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded_logx(x_plot,data1_mean,data1_std,ax_handle)
    p2=plot_shaded_logx(x_plot,data2_mean,data2_std,ax_handle)
    p3=plot_shaded_logx(x_plot,data3_mean,data3_std,ax_handle)
    p4=plot_shaded_logx(x_plot,data4_mean,data4_std,ax_handle)
    p5=plot_shaded_logx(x_plot,data5_mean,data5_std,ax_handle)
    p6=plot_shaded_logx(x_plot,data6_mean,data6_std,ax_handle)
    ax_handle.legend([p1, p2, p3, p4, p5, p6], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join(fig_dir,filename)
        plt.savefig(my_path)

def plot_sixline_shaded_logx_different_plotx(x_plot1,data1_mean,data1_std,x_plot2,data2_mean,data2_std,x_plot3,data3_mean,data3_std,x_plot4,data4_mean,data4_std,x_plot5,data5_mean,data5_std,x_plot6,data6_mean,data6_std,xlabel,ylabel,legend=["ZO-AGP: q=5","ZO-semi-EGDA: q=5","ZO-AGP: q=20","ZO-semi-EGDA: q=20","FO-AGP","FO-No Poison"],loc='upper left',fig_dir = 'Results_figures',filename=None):
    # plt.figure()
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded_logx(x_plot1,data1_mean,data1_std,ax_handle)
    p2=plot_shaded_logx(x_plot2,data2_mean,data2_std,ax_handle)
    p3=plot_shaded_logx(x_plot3,data3_mean,data3_std,ax_handle)
    p4=plot_shaded_logx(x_plot4,data4_mean,data4_std,ax_handle)
    p5=plot_shaded_logx(x_plot5,data5_mean,data5_std,ax_handle)
    p6=plot_shaded_logx(x_plot6,data6_mean,data6_std,ax_handle)
    ax_handle.legend([p1, p2, p3, p4, p5, p6], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join(fig_dir,filename)
        plt.savefig(my_path)

def plot_fourline_shaded_different_plotx(x_plot1,data1_mean,data1_std,color1,x_plot2,data2_mean,data2_std,color2,x_plot3,data3_mean,data3_std,color3,x_plot4,data4_mean,data4_std,color4,xlabel,ylabel,legend=["ZO-AGP: q=5","ZO-semi-EGDA: q=5","ZO-AGP: q=20","ZO-semi-EGDA: q=20"],loc='upper left',fig_dir = 'Results_figures',filename=None):
    # plt.figure()
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded(x_plot1,data1_mean,data1_std,ax_handle,color1)
    p2=plot_shaded(x_plot2,data2_mean,data2_std,ax_handle,color2)
    p3=plot_shaded(x_plot3,data3_mean,data3_std,ax_handle,color3)
    p4=plot_shaded(x_plot4,data4_mean,data4_std,ax_handle,color4)
    ax_handle.legend([p1, p2, p3, p4], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join(fig_dir,filename)
        plt.savefig(my_path)

def plot_twoline_shaded_logx(x_plot, data1_mean,data1_std,data2_mean,data2_std,xlabel,ylabel,legend,loc='upper left',filename=None):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded_logx(x_plot, data1_mean,data1_std,ax_handle)
    p2=plot_shaded_logx(x_plot, data2_mean,data2_std,ax_handle)
    ax_handle.legend([p1, p2], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)


def plot_twoline_shaded(x_plot, data1_mean, data1_std, data2_mean, data2_std, xlabel, ylabel, legend,
                             loc='upper left', filename=None):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1 = plot_shaded(x_plot, data1_mean, data1_std, ax_handle)
    p2 = plot_shaded(x_plot, data2_mean, data2_std, ax_handle)
    ax_handle.legend([p1, p2], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename != None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)
        # plt.savefig(filename)
    # plt.show()
    # plt.close()

def plot_threeline_shaded(x_plot, data1_mean, data1_std, data2_mean, data2_std, data3_mean, data3_std, xlabel, ylabel, legend,
                             loc='upper left', filename=None):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1 = plot_shaded(x_plot, data1_mean, data1_std, ax_handle)
    p2 = plot_shaded(x_plot, data2_mean, data2_std, ax_handle)
    p3 = plot_shaded(x_plot, data3_mean, data3_std, ax_handle)
    ax_handle.legend([p1, p2, p3], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename != None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)
        # plt.savefig(filename)
    # plt.show()
    # plt.close()
def plot_fourline_shaded(x_plot, data1_mean, data1_std, data2_mean, data2_std, data3_mean, data3_std, data4_mean, data4_std, xlabel, ylabel, legend,
                             loc='upper left', filename=None):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1 = plot_shaded(x_plot, data1_mean, data1_std, ax_handle)
    p2 = plot_shaded(x_plot, data2_mean, data2_std, ax_handle)
    p3 = plot_shaded(x_plot, data3_mean, data3_std, ax_handle)
    p4 = plot_shaded(x_plot, data4_mean, data4_std, ax_handle)
    ax_handle.legend([p1, p2, p3, p4], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename != None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)

def multiplot_all_logx_updated_AGP_and_semi_EGDA(train_poison_data, train_clean_data,test_data,lambda_x,alpha,beta,q_vec, times,epsilon, idx_coord_plot, Result_path, fig_dir, filename_temp):

    #### multi-random trials & multi-q
    num_q = len(q_vec)

    atrl_AGP=np.zeros((len(idx_coord_plot),times,num_q))
    atrc_AGP=np.zeros((len(idx_coord_plot),times,num_q))
    atec_AGP=np.zeros((len(idx_coord_plot),times,num_q))
    sc_AGP=np.zeros((len(idx_coord_plot),times,num_q))
    ZO_AGP_time = np.zeros((num_q,len(idx_coord_plot)))

    atrl_semi_EG=np.zeros((len(idx_coord_plot),times,num_q))
    atrc_semi_EG=np.zeros((len(idx_coord_plot),times,num_q))
    atec_semi_EG=np.zeros((len(idx_coord_plot),times,num_q))
    sc_semi_EG=np.zeros((len(idx_coord_plot),times,num_q))
    ZO_semi_EGDA_time = np.zeros((num_q,len(idx_coord_plot)))

    atrl_FO=np.zeros((len(idx_coord_plot),times))
    atrc_FO=np.zeros((len(idx_coord_plot),times))
    atec_FO=np.zeros((len(idx_coord_plot),times))
    sc_FO=np.zeros((len(idx_coord_plot),times))

    atrl_BL=np.zeros((len(idx_coord_plot),times))
    atrc_BL=np.zeros((len(idx_coord_plot),times))
    atec_BL=np.zeros((len(idx_coord_plot),times))

    # atrl_T_AG=np.zeros((times,num_q))
    # # atel_AG=[]
    # atrc_T_AG=np.zeros((times,num_q))
    # atec_T_AG=np.zeros((times,num_q))
    # sc_T_AG=np.zeros((times,num_q))
    # D=np.shape(train_clean_data[0])[0]-1

   #### multiple trials and multiple random direction vectors
    for i in range(0,times):
        print("Evaluation for trial %d" % i)

        ### ZO-AGP
        for iq in range(0,len(q_vec)):
            q_rand = q_vec[iq]
            # filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j)+"_AGP"
            filename = filename_temp +"_q_" + str(q_rand) + "_exp_" + str(i) + "_AGP"  ### "lambda_"+str(lambda_x[i])+"_exp_"+str(j
            AG =np.load(Result_path + "/" + "ZO_AGP_results"+filename+".npz")
            AG_iter_res=AG['AG_iter_res'][idx_coord_plot,:] ### only evaluate the points of interests
            ZO_AGP_time[iq,:]=AG['AG_time'][idx_coord_plot]
            ###### do specific evaluations for this specific q
            all_train_loss_AG = all_loss(AG_iter_res, lambda_x, train_poison_data, train_clean_data, test_data) ## F(x^t, \delta^t)
            stat_con_AG = stationary_condition(AG_iter_res, train_poison_data, train_clean_data, lambda_x=lambda_x,
                                                   alpha=alpha, beta=beta, epsilon=epsilon)
            ###### retrain under best poisoned vector
            all_train_accuracy_AG, all_test_accuracy_AG = all_acc_retrain(AG_iter_res, train_poison_data,
                                                                      train_clean_data,test_data, lambda_x)
            # print("all_test_accuracy_AG:", AG_iter_res)
            ### ZO-AGP
            atrl_AGP[:,i,iq] = all_train_loss_AG
            atrc_AGP[:,i,iq] = all_train_accuracy_AG
            atec_AGP[:,i,iq] = all_test_accuracy_AG
            sc_AGP[:,i,iq] = stat_con_AG

        ### ZO-Semi-EGDA
        for iq in range(0,len(q_vec)):
            q_rand = q_vec[iq]
            # filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j)+"_semi_EG"
            filename = filename_temp +"_q_" + str(q_rand) + "_exp_" + str(i) + "_semi_EG"  ### "lambda_"+str(lambda_x[i])+"_exp_"+str(j
            AG_EG =np.load(Result_path + "/" + "ZO_GDEGA_results_"+filename+".npz")
            AG_EG_iter_res=AG_EG['AG_iter_res'][idx_coord_plot,:] ### only evaluate the points of interests
            ZO_semi_EGDA_time[iq,:]=AG['AG_time'][idx_coord_plot]
            ###### do specific evaluations for this specific q
            all_train_loss_AG_EG = all_loss(AG_EG_iter_res, lambda_x, train_poison_data, train_clean_data, test_data) ## F(x^t, \delta^t)
            stat_con_AG_EG = stationary_condition(AG_EG_iter_res, train_poison_data, train_clean_data, lambda_x=lambda_x,
                                                   alpha=alpha, beta=beta, epsilon=epsilon)
            ###### retrain under best poisoned vector
            all_train_accuracy_AG_EG, all_test_accuracy_AG_EG = all_acc_retrain(AG_EG_iter_res, train_poison_data,
                                                                      train_clean_data,test_data, lambda_x)
            # print("all_test_accuracy_AG:", AG_EG_iter_res)
            ### ZO-semi-EGDA
            atrl_semi_EG[:,i,iq] = all_train_loss_AG_EG
            atrc_semi_EG[:,i,iq] = all_train_accuracy_AG_EG
            atec_semi_EG[:,i,iq] = all_test_accuracy_AG_EG
            sc_semi_EG[:,i,iq] = stat_con_AG_EG

        ### first-order case
        # filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j)+"_AGP"
        filename = filename_temp + "_exp_" + str(i) + "_AGP"

        FO=np.load(Result_path + "/" + "FOAG_4_2_SL_" + filename + ".npz")
        FO_iter_res=FO['FO_iter_res'][idx_coord_plot,:]
        FO_time=FO['FO_time'][idx_coord_plot]
        all_train_loss_FO=all_loss(FO_iter_res,lambda_x,train_poison_data, train_clean_data,test_data)
        all_train_accuracy_FO,all_test_accuracy_FO=all_acc_retrain(FO_iter_res,train_poison_data, train_clean_data,test_data,lambda_x)
        stat_con_FO=stationary_condition(FO_iter_res,train_poison_data, train_clean_data,lambda_x=lambda_x,alpha=alpha,beta=beta,epsilon=epsilon)
        atrl_FO[:, i] = all_train_loss_FO
        atrc_FO[:, i] = all_train_accuracy_FO
        atec_FO[:, i] = all_test_accuracy_FO
        sc_FO[:, i] = stat_con_FO

        ### no adversarial case
        # filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j)
        filename = filename_temp + "_exp_" + str(i)
        BL=np.load(Result_path + "/" + "FOnonAdv_4_2_SL_" + filename + ".npz")
        BL_iter_res=BL['FO_nonAdv_iter_res'][idx_coord_plot,:]
        BL_time=BL['FO_nonAdv_time'][idx_coord_plot]
        all_train_loss_BL =all_loss(BL_iter_res,lambda_x,train_poison_data, train_clean_data,test_data)
        all_train_accuracy_BL,all_test_accuracy_BL=all_acc(BL_iter_res,train_poison_data, train_clean_data,test_data)

        atrl_BL[:, i] = all_train_loss_BL
        atrc_BL[:, i] = all_train_accuracy_BL
        atec_BL[:, i] = all_test_accuracy_BL
        print("Ending Evaluation for trial %d" % i)

    # filename_plot = "lambda_" + str(lambda_x) + "_exp"
    filename_plot = "lambda_" + str(int(lambda_x*1000)) + "_exp"

    #### objective value versus iterations
    filename_plot_q = "lambda_" + str(int(lambda_x*1000)) + "_exp"

    plot_sixline_shaded_logx(np.array(idx_coord_plot)+1,
                            np.mean(atrl_AGP[:,:,0],axis=1), np.std(atrl_AGP[:,:,0],axis=1),
                                np.mean(atrl_semi_EG[:,:,0],axis=1), np.std(atrl_semi_EG[:,:,0],axis=1),
                                np.mean(atrl_AGP[:,:,1],axis=1), np.std(atrl_AGP[:,:,1],axis=1),
                                np.mean(atrl_semi_EG[:,:,1],axis=1), np.std(atrl_semi_EG[:,:,1],axis=1),
                                np.mean(atrl_FO, axis=1), np.std(atrl_FO, axis=1),
                                np.mean(atrl_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)),0*np.ones(len(idx_coord_plot)),
                                "Number of iterations", "Objective value", legend=["ZO-AGP: q=5", "ZO-GDEGA: q=5", "ZO-AGP: q=20", "ZO-GDEGA: q=20", "FO-AGP", "No Poison"],
                                loc='best', fig_dir = fig_dir, filename="train_loss_shaded_SL" + filename_plot_q + ".pdf")

    plot_sixline_shaded_logx(np.array(idx_coord_plot)+1, 
                                np.mean(atrc_AGP[:,:,0],axis=1), np.std(atrc_AGP[:,:,0],axis=1),
                                np.mean(atrc_semi_EG[:,:,0],axis=1), np.std(atrc_semi_EG[:,:,0],axis=1),
                                np.mean(atrc_AGP[:,:,1],axis=1), np.std(atrc_AGP[:,:,1],axis=1),
                                np.mean(atrc_semi_EG[:,:,1],axis=1), np.std(atrc_semi_EG[:,:,1],axis=1),
                                np.mean(atrc_FO, axis=1), np.std(atrc_FO, axis=1),
                                np.mean(atrc_BL, axis=1)[-1] * np.ones(len(idx_coord_plot)),0*np.ones(len(idx_coord_plot)),
                                "Number of iterations", "Training accuracy", legend=["ZO-AGP: q=5", "ZO-GDEGA: q=5", "ZO-AGP: q=20", "ZO-GDEGA: q=20", "FO-AGP", "No Poison"],
                                loc='best', fig_dir = fig_dir, filename="train_accuracy_shaded_SL" + filename_plot_q + ".pdf")

    plot_sixline_shaded_logx(np.array(idx_coord_plot)+1, 
                                np.mean(atec_AGP[:,:,0],axis=1), np.std(atec_AGP[:,:,0],axis=1),
                                np.mean(atec_semi_EG[:,:,0],axis=1), np.std(atec_semi_EG[:,:,0],axis=1),
                                np.mean(atec_AGP[:,:,1],axis=1), np.std(atec_AGP[:,:,1],axis=1),
                                np.mean(atec_semi_EG[:,:,1],axis=1), np.std(atec_semi_EG[:,:,1],axis=1),
                                np.mean(atec_FO, axis=1), np.std(atec_FO, axis=1),
                                np.mean(atec_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)), 0*np.ones(len(idx_coord_plot)),
                                "Number of iterations", "Testing accuracy", legend=["ZO-AGP: q=5", "ZO-GDEGA: q=5", "ZO-AGP: q=20", "ZO-GDEGA: q=20", "FO-AGP", "No Poison"],
                                loc='best', fig_dir = fig_dir, filename="test_accuracy_shaded_SL_iterations" + filename_plot_q + ".pdf")

    plot_sixline_shaded_logx_different_plotx(ZO_AGP_time[0,:], 
                                np.mean(atec_AGP[:,:,0],axis=1), np.std(atec_AGP[:,:,0],axis=1),
                                ZO_semi_EGDA_time[0,:],
                                np.mean(atec_semi_EG[:,:,0],axis=1), np.std(atec_semi_EG[:,:,0],axis=1),
                                ZO_AGP_time[1,:],
                                np.mean(atec_AGP[:,:,1],axis=1), np.std(atec_AGP[:,:,1],axis=1),
                                ZO_semi_EGDA_time[1,:],
                                np.mean(atec_semi_EG[:,:,1],axis=1), np.std(atec_semi_EG[:,:,1],axis=1),
                                FO_time,
                                np.mean(atec_FO, axis=1), np.std(atec_FO, axis=1),
                                BL_time,
                                np.mean(atec_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)), 0*np.ones(len(idx_coord_plot)),
                                "CPU time (seconds)", "Testing accuracy", legend=["ZO-AGP: q=5", "ZO-GDEGA: q=5", "ZO-AGP: q=20", "ZO-GDEGA: q=20", "FO-AGP", "No Poison"],
                                loc='best', fig_dir = fig_dir, filename="test_accuracy_shaded_SL_CPU_time" + filename_plot_q + ".pdf")
    
    plot_fiveline_shaded_logx(np.array(idx_coord_plot)+1,
                                np.mean(sc_AGP[:,:,0],axis=1), np.std(sc_AGP[:,:,0],axis=1),
                                np.mean(sc_semi_EG[:,:,0],axis=1), np.std(sc_semi_EG[:,:,0],axis=1),
                                np.mean(sc_AGP[:,:,1],axis=1), np.std(sc_AGP[:,:,1],axis=1),
                                np.mean(sc_semi_EG[:,:,1],axis=1), np.std(sc_semi_EG[:,:,1],axis=1),
                                np.mean(sc_FO, axis=1), np.std(sc_FO, axis=1),
                    "Number of iterations", "Stationary gap",legend=["ZO-AGP: q=5", "ZO-GDEGA: q=5", "ZO-AGP: q=20", "ZO-GDEGA: q=20", "FO-AGP"],loc='upper left',
                                fig_dir = fig_dir, filename="stationary_condition_shaded_SL"+filename_plot_q+".pdf")

def multiplot_all_logx_updated_AGP_and_semi_EGDA_no_FO(train_poison_data, train_clean_data,test_data,lambda_x,alpha,beta,q_vec, times,epsilon, idx_coord_plot, Result_path, fig_dir, filename_temp):

    #### multi-random trials & multi-q
    num_q = len(q_vec)
    # you can choose 0 for plotting our figure 3, or you choose 1 for plotting your figure
    flag_data_to_results = 1 
    if flag_data_to_results:
        atrl_AGP=np.zeros((len(idx_coord_plot),times,num_q))
        atrc_AGP=np.zeros((len(idx_coord_plot),times,num_q))
        atec_AGP=np.zeros((len(idx_coord_plot),times,num_q))
        sc_AGP=np.zeros((len(idx_coord_plot),times,num_q))
        ZO_AGP_time = np.zeros((num_q,len(idx_coord_plot)))

        atrl_semi_EG=np.zeros((len(idx_coord_plot),times,num_q))
        atrc_semi_EG=np.zeros((len(idx_coord_plot),times,num_q))
        atec_semi_EG=np.zeros((len(idx_coord_plot),times,num_q))
        sc_semi_EG=np.zeros((len(idx_coord_plot),times,num_q))
        ZO_semi_EGDA_time = np.zeros((num_q,len(idx_coord_plot)))

        atrl_BL=np.zeros((len(idx_coord_plot),times))
        atrc_BL=np.zeros((len(idx_coord_plot),times))
        atec_BL=np.zeros((len(idx_coord_plot),times))

    #### multiple trials and multiple random direction vectors
        for i in range(0,times):
            print("Evaluation for trial %d" % i)

            ### ZO-AGP
            for iq in range(0,len(q_vec)):
                q_rand = q_vec[iq]
                # filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j)+"_AGP"
                filename = filename_temp +"_q_" + str(q_rand) + "_exp_" + str(i) + "_AGP"  ### "lambda_"+str(lambda_x[i])+"_exp_"+str(j
                AG =np.load(Result_path + "/" + "ZO_AGP_results_"+filename+".npz")
                AG_iter_res=AG['AG_iter_res'][idx_coord_plot,:] ### only evaluate the points of interests
                ZO_AGP_time[iq,:]=AG['AG_time'][idx_coord_plot]
                ###### do specific evaluations for this specific q
                all_train_loss_AG = all_loss(AG_iter_res, lambda_x, train_poison_data, train_clean_data, test_data) ## F(x^t, \delta^t)
                stat_con_AG = stationary_condition(AG_iter_res, train_poison_data, train_clean_data, lambda_x=lambda_x,
                                                    alpha=alpha, beta=beta, epsilon=epsilon)
                ###### retrain under best poisoned vector
                all_train_accuracy_AG, all_test_accuracy_AG = all_acc_retrain(AG_iter_res, train_poison_data,
                                                                        train_clean_data,test_data, lambda_x)
                # print("all_test_accuracy_AG:", AG_iter_res)
                ### ZO-AGP
                atrl_AGP[:,i,iq] = all_train_loss_AG
                atrc_AGP[:,i,iq] = all_train_accuracy_AG
                atec_AGP[:,i,iq] = all_test_accuracy_AG
                sc_AGP[:,i,iq] = stat_con_AG
            
            ### ZO-GDEGA
            for iq in range(0,len(q_vec)):
                q_rand = q_vec[iq]
                # filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j)+"_semi_EG"
                filename = filename_temp +"_q_" + str(q_rand) + "_exp_" + str(i) + "_GDEGA"  ### "lambda_"+str(lambda_x[i])+"_exp_"+str(j
                AG_EG =np.load(Result_path + "/" + "ZO_GDEGA_results_"+filename+".npz")
                AG_EG_iter_res=AG_EG['AG_iter_res'][idx_coord_plot,:] ### only evaluate the points of interests
                ZO_semi_EGDA_time[iq,:]=AG['AG_time'][idx_coord_plot]
                ###### do specific evaluations for this specific q
                all_train_loss_AG_EG = all_loss(AG_EG_iter_res, lambda_x, train_poison_data, train_clean_data, test_data) ## F(x^t, \delta^t)
                stat_con_AG_EG = stationary_condition(AG_EG_iter_res, train_poison_data, train_clean_data, lambda_x=lambda_x,
                                                    alpha=alpha, beta=beta, epsilon=epsilon)
                ###### retrain under best poisoned vector
                all_train_accuracy_AG_EG, all_test_accuracy_AG_EG = all_acc_retrain(AG_EG_iter_res, train_poison_data,
                                                                        train_clean_data,test_data, lambda_x)
                # print("all_test_accuracy_AG:", AG_EG_iter_res)
                ### ZO-semi-EGDA
                atrl_semi_EG[:,i,iq] = all_train_loss_AG_EG
                atrc_semi_EG[:,i,iq] = all_train_accuracy_AG_EG
                atec_semi_EG[:,i,iq] = all_test_accuracy_AG_EG
                sc_semi_EG[:,i,iq] = stat_con_AG_EG
            
            ### no adversarial case
            # filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j)
            filename = filename_temp + "_exp_" + str(i)
            BL=np.load(Result_path + "/" + "FOnonAdv_4_2_SL_" + filename + ".npz")
            BL_iter_res=BL['FO_nonAdv_iter_res'][idx_coord_plot,:]
            BL_time=BL['FO_nonAdv_time'][idx_coord_plot]
            all_train_loss_BL =all_loss(BL_iter_res,lambda_x,train_poison_data, train_clean_data,test_data)
            all_train_accuracy_BL,all_test_accuracy_BL=all_acc(BL_iter_res,train_poison_data, train_clean_data,test_data)

            atrl_BL[:, i] = all_train_loss_BL
            atrc_BL[:, i] = all_train_accuracy_BL
            atec_BL[:, i] = all_test_accuracy_BL
            
            print("Ending Evaluation for trial %d" % i)
        # save plot data
        np.savez(Result_path + "/" +"ZO_AGP" + "_plot_data" + ".npz", atrl_AGP=atrl_AGP, atrc_AGP=atrc_AGP, atec_AGP=atec_AGP, sc_AGP=sc_AGP, ZO_AGP_time = ZO_AGP_time)
        np.savez(Result_path + "/" +"ZO_GDEGA" + "_plot_data" + ".npz", atrl_semi_EG=atrl_semi_EG, atrc_semi_EG=atrc_semi_EG, atec_semi_EG=atec_semi_EG, sc_semi_EG=sc_semi_EG, ZO_semi_EGDA_time=ZO_semi_EGDA_time)
        np.savez(Result_path + "/" + "FOnonAdv" + "_plot_data" + ".npz", atrl_BL=atrl_BL, atrc_BL=atrc_BL, atec_BL=atec_BL, BL_time= BL_time)

    # read plot data
    filename_plot = "lambda_" + str(int(lambda_x*1000)) + "_exp"
    AGP_results = np.load(Result_path + "/" +"ZO_AGP" + "_plot_data" + ".npz")
    atrl_AGP = AGP_results['atrl_AGP']
    atrc_AGP = AGP_results['atrc_AGP']
    atec_AGP = AGP_results['atec_AGP']
    sc_AGP = AGP_results['sc_AGP']
    ZO_AGP_time = AGP_results['ZO_AGP_time']
    EG_results = np.load(Result_path + "/" +"ZO_GDEGA" + "_plot_data" + ".npz")
    atrl_semi_EG = EG_results['atrl_semi_EG']
    atrc_semi_EG = EG_results['atrc_semi_EG']
    atec_semi_EG = EG_results['atec_semi_EG']
    sc_semi_EG = EG_results['sc_semi_EG']
    ZO_semi_EGDA_time = EG_results['ZO_semi_EGDA_time']
    FOnonAdv_results = np.load(Result_path + "/" +"FOnonAdv" + "_plot_data" + ".npz")
    atrl_BL = FOnonAdv_results['atrl_BL']
    atrc_BL = FOnonAdv_results['atrc_BL']
    atec_BL = FOnonAdv_results['atec_BL']
    BL_time = FOnonAdv_results['BL_time']
    #### objective value versus iterations
    #for iq in range(0, len(q_vec)):
    filename_plot_q = "lambda_" + str(int(lambda_x*1000)) + "_exp"

    plot_fiveline_shaded_logx(np.array(idx_coord_plot)+1, 
                                np.mean(atrc_BL, axis=1)[-1] * np.ones(len(idx_coord_plot)),0*np.ones(len(idx_coord_plot)), "purple",
                                np.mean(atrc_AGP[:,:,0],axis=1), np.std(atrc_AGP[:,:,0],axis=1), "royalblue", 
                                np.mean(atrc_semi_EG[:,:,0],axis=1), np.std(atrc_semi_EG[:,:,0],axis=1), "orange",
                                np.mean(atrc_AGP[:,:,1],axis=1), np.std(atrc_AGP[:,:,1],axis=1), "green",
                                np.mean(atrc_semi_EG[:,:,1],axis=1), np.std(atrc_semi_EG[:,:,1],axis=1), "red",
                                "Number of iterations", "Training accuracy", legend=["No Poison", "ZO-AGP: q=5", "ZO-GDEGA: q=5 (Ours)", "ZO-AGP: q=20", "ZO-GDEGA: q=20 (Ours)"],
                                loc='best', fig_dir = fig_dir, filename="train_accuracy_shaded_SL" + filename_plot_q + ".pdf")

    plot_fiveline_shaded_logx(np.array(idx_coord_plot)+1, 
                                np.mean(atec_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)), 0*np.ones(len(idx_coord_plot)), "purple",
                                np.mean(atec_AGP[:,:,0],axis=1), np.std(atec_AGP[:,:,0],axis=1), "royalblue", 
                                np.mean(atec_semi_EG[:,:,0],axis=1), np.std(atec_semi_EG[:,:,0],axis=1), "orange",
                                np.mean(atec_AGP[:,:,1],axis=1), np.std(atec_AGP[:,:,1],axis=1), "green",
                                np.mean(atec_semi_EG[:,:,1],axis=1), np.std(atec_semi_EG[:,:,1],axis=1), "red",
                                "Number of iterations", "Testing accuracy", legend=["No Poison", "ZO-AGP: q=5", "ZO-GDEGA: q=5 (Ours)", "ZO-AGP: q=20", "ZO-GDEGA: q=20 (Ours)"],
                                loc='best', fig_dir = fig_dir, filename="test_accuracy_shaded_SL_iterations" + filename_plot_q + ".pdf")

    end = 96 # epsilon_test
    plot_fourline_shaded_different_plotx(ZO_AGP_time[0,:]-ZO_AGP_time[0,0], 
                                np.mean(atec_AGP[:,:,0],axis=1), np.std(atec_AGP[:,:,0],axis=1), "royalblue",
                                ZO_semi_EGDA_time[0,:end]-ZO_semi_EGDA_time[0,0],
                                np.mean(atec_semi_EG[:end,:,0],axis=1), np.std(atec_semi_EG[:end,:,0],axis=1), "orange",
                                ZO_AGP_time[1,:end]-ZO_AGP_time[1,0],
                                np.mean(atec_AGP[:end,:,1],axis=1), np.std(atec_AGP[:end,:,1],axis=1), "green",
                                ZO_semi_EGDA_time[1,:end]-ZO_semi_EGDA_time[1,0],
                                np.mean(atec_semi_EG[:end,:,1],axis=1), np.std(atec_semi_EG[:end,:,1],axis=1), "red",
                                "CPU time (seconds)", "Testing accuracy", legend=["ZO-AGP: q=5", "ZO-GDEGA: q=5 (Ours)", "ZO-AGP: q=20", "ZO-GDEGA: q=20 (Ours)"],
                                loc='best', fig_dir = fig_dir, filename="test_accuracy_shaded_SL_CPU_time" + filename_plot_q + ".pdf")

########################################################################################
######################## Main function to run all algorithms and plots #############################
########################################################################################
if __name__=="__main__":
    train_ratio = 0.7
    sigma2 = 1
    noise_std = 1e-3
    n_iters = 50000
    
    
    ############## Part I: data generation   ######################
    flag_dataGeneration = 0 # synthetic data flag
    flag_dataLoad = 1
    flag_synthetic = 0
    dataset_name = 'epsilon_test_mean_std'
    if flag_synthetic:
        n_tr = 1000
        D_x = 100
        x_gt = 1 * np.ones(D_x)
        batch_sz = 100
        lr_x = 0.05
        lr_delta = 0.02
        Result_path = r"D:\ZO-minimax-data\NC_C_results"
        if not os.path.exists(os.path.dirname(os.path.realpath(__file__))+"/Results_figures"):
            os.makedirs(os.path.dirname(os.path.realpath(__file__))+"/Results_figures")
    else:
        if dataset_name == 'epsilon_test_mean_std':
            poison_ratio = 0.1
            n_tr = 35000
            D_x = 2000
            batch_sz = 10
            batch_sz_poison = 10
            lr_x = 0.05
            lr_delta = 0.02
            eps_perturb = 2
            mu1 = 1.0/n_iters
            mu2 = 1.0/n_iters
            step = [mu1, mu2]
            Result_path = r"./NC_C_real_data_epsilon_test_results_10_trials"
            if not os.path.exists(Result_path):
                os.makedirs(Result_path) 

            if not os.path.exists(os.path.dirname(os.path.realpath(__file__))+"/Real_data_epsilon_test_mean_std_Results_figures"):
                os.makedirs(os.path.dirname(os.path.realpath(__file__))+"/Real_data_epsilon_test_mean_std_Results_figures")
        if dataset_name == 'HIGGS_mean_std':
            poison_ratio = 0.1
            n_tr = 7700000
            D_x = 28
            ## choice 1
            batch_sz = 1024
            batch_sz_poison = 1024
            ## choice 2
            # batch_sz = 512
            # batch_sz_poison = 512
            lr_x = 0.05
            lr_delta = 0.02 ### 0.02
            eps_perturb = 2
            mu1 = 0.001
            mu2 = 0.001
            step = [mu1, mu2]
            Result_path = r"D:\ZO-minimax-data\NC_C_real_data_HIGGS_mean_std_poison_ratio0.05_results_5_trials"
            if not os.path.exists(Result_path):
                os.makedirs(Result_path)
            if not os.path.exists(os.path.dirname(os.path.realpath(__file__))+"/Real_data_HIGGS_mean_std_poison_ratio0.05_Results_figures"):
                os.makedirs(os.path.dirname(os.path.realpath(__file__))+"/Real_data_HIGGS_mean_std_poison_ratio0.05_Results_figures")

        x_gt = 1 * np.ones(D_x)
        
        
    if flag_synthetic: 
        Data_filename = "D_4_2_SL"
        Data_path = "D:\ZO-minimax-data" + "\data_" + str(n_tr)
        if not os.path.exists(Data_path):
            os.makedirs(Data_path)
        if not os.path.exists(Result_path):
            os.makedirs(Result_path)
        if flag_dataGeneration:
            generate_data(n_tr, sigma2, x_gt, noise_std, Data_path, Data_filename)
            train_data, train_poison_data, train_clean_data, test_data = generate_train_and_test_data(train_ratio, poison_ratio, Data_path,  Data_filename, True)
            index = generate_index(np.size(train_clean_data,0), batch_sz, n_iters, Data_path, Data_filename)
            index_batch = generate_index(np.size(train_clean_data,0), int(round(1*np.size(train_clean_data,0))),
                                        n_iters, Data_path, Data_filename+"batch")
            
        else:
            if flag_dataLoad:
                train_data, train_poison_data, train_clean_data, test_data = load_train_and_test_data(poison_ratio,Data_path,Data_filename)
                index = load_index(Data_path, Data_filename)
                index_batch = load_index(Data_path, Data_filename+"batch")
                print("train_data:", train_data.shape) # 700×(100+1)
                print("train_poison_data:", train_poison_data.shape) # 70×(100+1)
                print("train_clean_data:", train_clean_data.shape) # 630×(100+1)
                print("test_data:", test_data.shape) # 300×(100+1)
                print("index:", index.shape) # 50000 × batch_size: 100
    else:
        if not os.path.exists(Result_path):
            os.makedirs(Result_path)
        train_data, train_poison_data, train_clean_data, test_data = load_real_train_and_test_data(train_ratio, poison_ratio, dataset_name)
        print("train_data:", train_data.shape) # 700×(100+1)
        print("train_poison_data:", train_poison_data.shape) # 70×(100+1)
        print("train_clean_data:", train_clean_data.shape) # 630×(100+1)
        print("test_data:", test_data.shape) # 300×(100+1)
        index=[]
        index_poison=[]
        for i in range(0, n_iters):
            temp=np.array(random.sample(range(0,np.size(train_clean_data,0)),batch_sz)) ### mini-batch scheme
            index.append(temp)  #### generate a list
            temp=np.array(random.sample(range(0,np.size(train_poison_data,0)),batch_sz_poison)) ### mini-batch scheme
            index_poison.append(temp)  #### generate a list

    ############## Part II: poisoning attack learning for different q, lambda, multiple trials ###############
    flag_train = 1
    lambda_x =  [1e-3]
    q_vec = [5,20] #multiple random direction vectors
    n_trials = 10


    if flag_train:
        for i in range(0,len(lambda_x)):
            for j in range(0,n_trials):
                #### if attack loss is large, then the classifier is accurate and no adv.
                # x_ini = x_gt + 1*np.random.normal(0, 1, D_x) ### initial x from a point close to ground truth (no adv. is considered)
                x_ini_tmp = np.random.normal(0, 1, D_x)
                # x_ini = x_ini_tmp
                delta_ini_tmp = np.zeros(D_x) #np.random.normal(0, 0.01, D_x)  #np.random.normal(0, 0.1, D_x)  # np.zeros(D_x)  # np.random.uniform(-eps_perturb,eps_perturb,D_x) ### inital delta
                # delta_ini = delta_ini_tmp
                #### stationary condition for initial point
                thr_stat = 0
                for i_test in range(0,100):
                    iter_res = np.zeros((1,D_x+D_x))
                    iter_res[0,:D_x] = delta_ini_tmp
                    iter_res[0,D_x:] = x_ini_tmp
                    stat_temp = stationary_condition(iter_res, train_poison_data, train_clean_data, lambda_x[i], lr_delta, lr_x, eps_perturb)
                    if stat_temp > thr_stat:
                        thr_stat = stat_temp
                        x_ini = x_ini_tmp
                        delta_ini = delta_ini_tmp
                        print("Stationary condition = %4.3f" % stat_temp)
                    x_ini_tmp =  np.random.normal(0, 1, D_x) # np.random.uniform(-2, 2, D_x)
                    delta_ini_tmp = np.zeros(D_x) #np.random.uniform(-1, 1, D_x)  # np.random.normal(0, 0.01, D_x)  #np.random.normal(0, 0.1, D_x)  # np.zeros(D_x)  # np.random.uniform(-eps_perturb,eps_perturb,D_x) ### inital delta
                #### ZO-AGP
                #### AG_sol_track: first D_x dimension is x, and the last D_x dimension is delta
                for iq in range(0,len(q_vec)):
                    q_rand = q_vec[iq]
                    x_opt_AG_AGP, AG_AGP_sol_track, AG_AGP_time_track = AG_main_batch_SL_AGP(train_poison_data, train_clean_data,
                                    x_ini, delta_ini, eps_perturb, step, n_iters, x_gt, index, index_poison, lambda_x[i], lr_delta, lr_x, q_rand,
                                     Result_path, filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j)+"_AGP")
                    
                #### ZO-GDEGA
                #### AG_sol_track: first D_x dimension is x, and the last D_x dimension is delta
                for iq in range(0,len(q_vec)):
                    q_rand = q_vec[iq]
                    x_opt_AG_EG, AG_EG_sol_track, AG_EG_time_track = AG_main_batch_SL_EG(train_poison_data, train_clean_data,
                                    x_ini, delta_ini, eps_perturb, step, n_iters, x_gt, index, index_poison, lambda_x[i], lr_delta, lr_x, q_rand,
                                     Result_path, filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j)+"_GDEGA")

                if flag_synthetic:
                    #### FO-AGP
                    x_opt_FO, FO_sol_track, FO_time_track \
                        = FO_main_batch_SL(train_poison_data, train_clean_data, x_ini, delta_ini, eps_perturb, n_iters, x_gt,
                                        index, index_poison, lambda_x[i], lr_delta, lr_x,  Result_path,
                                        filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j) + "_AGP")

                #### No-adv effect
                x_opt_FO_nonAdv, FO_nonAdv_sol_track, FO_nonAdv_time_track = FO_nonAdv_main_batch_SL(train_poison_data, train_clean_data, x_ini,
                                                                                                     delta_ini, eps_perturb, n_iters, x_gt,
                                                                                                     index, index_poison, lambda_x[i], lr_delta, lr_x, Result_path,
                                                                                                     filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j))
    ################################# plot figures for different algorithms to solve NC-C problems ###############
    flag_plot = 1
    if flag_plot:
        if flag_synthetic:
            idx_coordinate_plot = range(0,n_iters,50)
            fig_dir = 'Results_figures'
            for i in range(0, len(lambda_x)):
                filename_temp = "lambda_" + str(lambda_x[i])
                multiplot_all_logx_updated_AGP_and_semi_EGDA(train_poison_data, train_clean_data, test_data, lambda_x[i], lr_delta,lr_x, q_vec,  n_trials, eps_perturb,idx_coordinate_plot, Result_path, fig_dir, filename_temp)
        else:
            idx_coordinate_plot = range(0,n_iters,500)
            fig_dir = 'Real_data_' + dataset_name + '_Results_figures'
            for i in range(0, len(lambda_x)):
                filename_temp = "lambda_" + str(lambda_x[i])
                multiplot_all_logx_updated_AGP_and_semi_EGDA_no_FO(train_poison_data, train_clean_data, test_data, lambda_x[i], lr_delta,lr_x, q_vec,  n_trials, eps_perturb,idx_coordinate_plot, Result_path, fig_dir, filename_temp)

    