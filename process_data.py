import os
import scipy.stats
import joblib
import pickle
import numpy as np
from multiprocessing import Pool
from core import data_utils
from core import utils

def box_based_outlier(x):
    percentile = np.percentile(x,[25,50,75])
    iqr = percentile[-1]-percentile[0]
    lp = percentile[0]-3*iqr
    up = percentile[-1]+3*iqr
    x[x<lp] = lp
    x[x>up] = up
    return x,(lp,up,iqr)

def preprocess(X_train,X_test,y_train,y_testtag="2017"):
    x_list = []
    x_test_list = []
    bound_dict = dict()
    if not os.path.exists(f"materials/boxoutdata_{tag}_100.pkl"):
        print('Stage 1')
        for i in range(0,X_train.shape[1]): 
            x,(lp,up,iqr) = box_based_outlier(X_train[:,i].copy())
            x_list.append(x.reshape(x.shape[0],1))
            x_t = X_test[:,i].copy()
            x_t[x_t<lp] = lp#if test<main but >lp
            x_t[x_t>up] = up 
            x_test_list.append(x_t.reshape(x_t.shape[0],1))
            print("Feature "+str(i)+" has "+str(len(set(X_train[:,i])))+" different values. After processing, "+str(len(set(x_t)))+" features left")
        x_train = np.concatenate(x_list,axis=1)
        x_test = np.concatenate(x_test_list,axis=1)

        print('Stage 2')
        valueset_list = []
        temp = []
        temp_test = []
        for i in range(0,x_train.shape[1]):
            #evenly cut it into 1000/100 sections
            if len(set(x_train[:,i]))>100:
                x,valueset = utils.process_column_evenly(x_train[:,i],100)
            else:
                x = x_train[:,i]
                valueset = sorted(list(set(x_train[:,i])))
                valueset.append(1e26)
            x_t = x_test[:,i].copy()
            for vi in range(len(valueset)-1): #redundant features are eliminated here
                x_t[(x_t>=valueset[vi])&(x_t<valueset[vi+1])]=valueset[vi]
            x_t[x_t<valueset[0]] = valueset[0]
            print("After processing, "+str(i)+" has "+str(len(set(x)))+" different values, and test set has "+str(len(set(x_t)))+" different values.")
            temp.append(x.reshape(-1,1))    
            temp_test.append(x_t.reshape(-1,1))
            valueset_list.append(valueset)
        up = x_train.max(axis=0)
        lp = x_train.min(axis=0)
        x_train_ = np.concatenate(temp,axis=1)
        x_test_ = np.concatenate(temp_test,axis=1)
        joblib.dump([up,lp,valueset_list],f"materials/materials_{tag}.pkl")
        joblib.dump([x_train_,x_test_,y_train,y_test],f"materials/boxoutdata_{tag}_100.pkl")
    #else:
    #    x_train,x_test,y_train,y_test = joblib.load(f"materials/boxoutdata_{tag}_100.pkl")

def entropy(y):
    p = y[y==0].shape[0]/y.shape[0]
    entropy = p*math.log2(p+1e-5)+(1-p)*math.log2(1-p+1e-5)
    return -entropy

def calc_ig(y,y1,y2):
    y_ent = entropy(y)
    y1_ent = entropy(y1)
    y2_ent = entropy(y2)
    ig = y_ent - (len(y1)/len(y))*y1_ent - (len(y2)/len(y))*y2_ent
    return ig

def kl_divergence(y1,y2):
    p = y1[y1==0].shape[0]/y1.shape[0]
    P = [p,1-p]
    q = y2[y2==0].shape[0]/y2.shape[0]
    Q = [q,1-q]
    #print(P,Q)
    return scipy.stats.entropy(P,Q)

def compress_values(values,y_train,value_test,f,valueset,threshold):
    """Compress the feature based on threshold
    :param values: the feature f of the training set
    :param y_train: the label of training set
    :param value_test: the feature f of the testing set
    :param f: the index of current feature
    :param gap: the gap for condensing the valueset
    :param threshold: the lower bound of density
    """
    valueset = list(valueset[:-1])
    if len(valueset)==1:
        return values.reshape(-1,1),value_test.reshape(-1,1),None,valueset
    #print("Feature "+str(f)+" has "+str(len(set(values)))+" different values")
    density_list = []
    rule = dict()
    #gap = threshold#valueset[1] - valueset[0]
    for index,m in enumerate(valueset):
        density = values[values==m].shape[0]/values.shape[0]
        density_list.append(density)
    while True:
        min_density = min(density_list)
        if min_density > threshold or len(density_list)<=1:
            break
        index = density_list.index(min_density)
        if index == 0:
            target_index = index+1
        elif index == len(valueset)-1:
            target_index = index-1
        else:
            l = y_train[values==valueset[index-1]]
            m = y_train[values==valueset[index]]
            u = y_train[values==valueset[index+1]]
            if l.shape[0]==0 or m.shape[0]==0:#if it is not a empty section
                target_index = index+1
            elif kl_divergence(m,l)<kl_divergence(m,u):
                target_index = index-1
            else:
                target_index = index+1
        main = valueset[target_index]
        sub = valueset[index]
        values[values==sub] = main
        rule[main] = rule.get(main,[])
        rule[main].append(sub)
        density_list[target_index] += density_list[index]
        del valueset[index]
        del density_list[index]
        #print("Merge value %.5f into %.5f"%(sub,main))
        if sub in rule:
            rule[main].extend(rule[sub])
            del rule[sub]
    #print("After processing, feature "+str(f)+" has "+str(len(valueset))+" features left.")
    for i in rule:
        for r in rule[i]:
            value_test[value_test==r] = i
    gap = 1/len(valueset)
    valueset = sorted(list(set(values)))
    #In case of override
    values_orig = values.copy()
    value_test_orig = value_test.copy()
    for i,j in enumerate(valueset):
        values[values_orig==j] = i*gap
        value_test[value_test_orig==j] = i*gap
    return values.reshape(-1,1),value_test.reshape(-1,1),rule,valueset

    
def combine(tag=2017):
    print("Stage 3")
    print("Multiprocessing")
    up, lp, valueset_list = joblib.load(f"materials/materials_{tag}.pkl")
    x_train,x_test,y_train,y_test = joblib.load(f"materials/boxoutdata_{tag}_100.pkl")
    up = x_train.max(axis=0)
    lp = x_train.min(axis=0)
    for threshold in [0.08]:#[0.01,0.02,0.04,0.08,0.16,0.32]:
        if os.path.exists("materials/compressed_%d_%d_material.pkl"%(tag,threshold*100)):
            print("X_train's sum:",x_train.sum()) 
            print("Threshold:",threshold)
            pool = Pool(processes=30)
            res_object = [pool.apply_async(compress_values,args=(x_train[:,i],y_train,x_test[:,i],i,valueset_list[i],threshold)) for i in range(x_train.shape[1])]
            res_train = [r.get()[0] for r in res_object]
            res_test = [r.get()[1] for r in res_object]
            res_rules = [r.get()[2] for r in res_object]
            res_valueset = [r.get()[3] for r in res_object]
            pool.close()
            pool.join()
            x_train_ = np.concatenate(res_train,axis=1)
            x_test_ = np.concatenate(res_test,axis=1)
            joblib.dump([x_train_,x_test_,y_train,y_test],"materials/compressed_%d_%d_reallocated.pkl"%(tag,threshold*100))
            joblib.dump([res_rules,res_valueset],"materials/compressed_%d_%d_material.pkl"%(tag,threshold*100))#used for processing other samples
            print(x_train_.sum()) 

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_utils.load_dataset('ember')
    preprocess(x_train, x_test, 2017)
    combine(2017)
    #x_train, y_train, emberdf = data_utils.load_dataset('emberall')
    #indices = (emberdf["appeared"]>"1971-01")&(emberdf["appeared"]<="2017-03")
    #preprocess(x_train[indices].copy(), x_train, y_train[indices].copy(),y_train, 03)
    #combine(03)
