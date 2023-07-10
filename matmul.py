#!/usr/bin/env python
# coding: utf-8

# ## Width Extension

# In[ ]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# ## GPU Device Check

# In[ ]:


from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())


# ## Import Packages

# In[ ]:


import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import numpy as np
import time
import tensorflow as tf
import pandas as pd
import random
import pandas as pd
import os
import gc


# ## TF_Session

# In[ ]:


# config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
# config.gpu_options.visible_device_list = "0"
# sess = tf.Session(config=config)
# sess.run(tf.global_variables_initializer())
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configuration)
sess.run(tf.global_variables_initializer())


# ## Function Definition

# In[ ]:


def sweep(x):
    div = x.sum(axis=1, dtype='float')
    res = x/div
    
    return res

def scale(y, c=True, sc=True):
    x = y.copy()

    if c:
        x -= x.mean()
    if sc and c:
        x /= x.std()
    elif sc:
        x /= np.sqrt(x.pow(2).sum().div(x.count() - 1))
    return x


# ## Load Dataset

# In[ ]:


readRDS = robjects.r['readRDS']
order_01_trimmed_adj_matrices_list = readRDS('all_adj_mtx_orders_list.rds')
# st_gsea_all_go = readRDS('/media/jaeseunglee/Storage/rstudio/rds/st_gsea_all_go.rds')
st_gsea_all_go_names = ["tb6761_aspp1","tb6761_igg","tb6804_aspp1","tb6804_igg","tb6824_aspp1","tb6824_igg","tb6859_aspp1","tb6859_igg"]


# ## Main

# In[ ]:


start = time.time()
print("======= START =======")
for a in list(range(len(st_gsea_all_go_names))):
    df = pd.read_csv('csv/' + st_gsea_all_go_names[a] + '.csv')
    for b in list(range(len(order_01_trimmed_adj_matrices_list.names))):
        if "O1_" + st_gsea_all_go_names[a] == order_01_trimmed_adj_matrices_list.names[b]:
#             print("Sample name is: " + st_gsea_all_go.names[a])
#             print("Adjacency matrix: " + order_01_trimmed_adj_matrices_list.names[b])
            pathways_name = list(np.concatenate(df[["Unnamed: 0"]].values.tolist()))
            pathways = np.asmatrix(df.drop(columns="Unnamed: 0").values)
            adj = np.float32(np.asmatrix(order_01_trimmed_adj_matrices_list[b]))
            adj2 = np.float32(sweep(adj))
            pl = len(pathways)
            row = list(range(pl))
            p_val = []
            for i in list(range(pl)):
                for j in list(range(pl)):
                    pathway_i = pathways_name[i]
                    pathway_j = pathways_name[j]
                    combo_name = pathway_i + '_x_' + pathway_j
#                     print("====================================================================================================================================")
#                     print("= Sample name: " + st_gsea_all_go_names[a])
#                     print("= Adjacency matrix: " + order_01_trimmed_adj_matrices_list.names[b])
#                     print("= Combo name: [ %s ]"%(combo_name))
#                     print("====================================================================================================================================")
                    obj1 = np.float32(scale(pathways[i,]).getT())
                    obj2 = np.float32(scale(pathways[j,]))
                    prod = tf.matmul(obj1, obj2)
                    final = tf.matmul(adj2,prod)
                    compare = []
                    num_row = list(range(len(obj1)))
#                     print("\n")
#                     print("================================================ Permutation Test ===================================================================")
#                     with tf.Session(config=config) as sess:
                    configuration = tf.compat.v1.ConfigProto()
                    configuration.gpu_options.allow_growth = True
                    sess = tf.compat.v1.Session(config=configuration)
                    sess.run(tf.global_variables_initializer())
                    for k in list(range(100)):
                        print(k+1)
                        obj1 = pd.DataFrame(obj1)
                        obj2 = pd.DataFrame(obj2)
                        rng = np.random.default_rng(seed = k)
                        new_order = rng.permutation(num_row,0)
                        x = np.float32(np.asmatrix(obj1.iloc[new_order,:]))
                        y = np.float32(np.asmatrix(obj2.iloc[:,new_order]))
                        prod_shuff = tf.tensordot(x, y, axes=1)
                        final_shuff = tf.matmul(adj2,prod_shuff)
                        scc_shuff = tf.reduce_mean(tf.linalg.tensor_diag_part(final_shuff),0)
                        compare.append(scc_shuff)
                        del scc_shuff
                        del final_shuff
                        del prod_shuff
                        del x 
                        del y
                        gc.collect()
#                         print("=======================================================================================================================================")
#                         print("\n")
                    del obj1
                    del obj2
                    
                    compare = sess.run(compare)
                    local_scc = sess.run(tf.linalg.tensor_diag_part(final))
                    local_scc_list = list(local_scc.round(3))
                    global_scc = np.float32(np.mean(local_scc))
                    p_val = ((compare > global_scc).sum()/100)
                    del final
                    del local_scc
                    gc.collect()
#                         print("========================================================================================================================================")
#                         print("local_scc:")
#                         print(local_scc)
#                         print("global_scc: %f"%(global_scc))
#                         print("p_val:")
#                         print(p_val)
#                         print("========================================================================================================================================")
#                         print("\n")
                        
                    sess.close()
                    tf.keras.backend.clear_session()
                    if i == 0 and j == 0:
                        df2 = 0
                        df2 = pd.DataFrame({"combo_name":[combo_name],'local_scc':[local_scc_list], 'global_scc':[global_scc],'p_val':[p_val]})
                        print(df2.memory_usage())
                    else:
                        df2.loc[len(df2.index)] = [combo_name, local_scc_list, global_scc,p_val] 
                        print(df2.memory_usage())
                    
                    del compare
                    del local_scc_list
                    del global_scc
                    del p_val 
                    gc.collect()
                    end = time.time()
                    elapsed = (end - start)
#                         print("%.2f sec"%(elapsed))
                    print('pathway progress: ' + str((i*pl+j+1)) + '/' + str((pl*pl)))
#                         print('[SAVED!] ' + st_gsea_all_go_names[a] + '_' + order_01_trimmed_adj_matrices_list.names[b] + '.csv')
                path = st_gsea_all_go_names[a] + '_' + order_01_trimmed_adj_matrices_list.names[b] 
                isExist = os.path.exists(path)
                if not isExist:
                   os.makedirs(path)
                df2.to_csv(path + '/' + st_gsea_all_go_names[a] + '_' + order_01_trimmed_adj_matrices_list.names[b] + '_' + str(i) + '.csv',index = None)
            df2.to_csv("Final_" + st_gsea_all_go_names[a] + '_' + order_01_trimmed_adj_matrices_list.names[b] + '.csv',index = None)


