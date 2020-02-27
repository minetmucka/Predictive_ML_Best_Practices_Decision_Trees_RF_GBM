#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
# # Loading Data into Dataframe
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import os
import math
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,make_scorer,precision_score,recall_score,roc_auc_score,f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp

df=pd.read_csv("titanic.csv",header=0)
df.head(10)

# ### Column Significance
#     PassengerId : Sequential Id assigned to the Passenger
#     Survived:
#     Pclass: Class of the Passenger
#     Name:
#     Sex:
#     Age:
#     SibSp:
#     Parch:
#     Ticket:
#     Fare:
#     Cabin:
#     Embarked: Source Port (C=Cherbourg,Q=Queenstown,S=Southampton)

# # Removing unwanted Columns : Name and ID attributes
df.drop(columns = ['PassengerId','Name','Ticket'],axis=1,inplace=True)
# # Creating new variables
df['Dependent_ind']='0'
df['Parent_child_ind']='0'
df['Sibling_Spouse_ind']='0'
df['Child_ind']='0'
df['Embarked_ind']='0'

df.loc[df['Parch']>0,'Dependent_ind']='1'
df.loc[df['SibSp']>0,'Dependent_ind']='1'
df.loc[df['Parch']>0,'Parent_child_ind']='1'
df.loc[df['SibSp']>0,'Sibling_Spouse_ind']='1'
df.loc[df['Age']<16,'Child_ind']='1'
df.loc[df['Embarked']=='C','Embarked_ind']='1'

df = df[['Survived','Pclass','Sex','Age','Child_ind','SibSp','Parch','Dependent_ind','Parent_child_ind','Sibling_Spouse_ind','Fare','Cabin','Embarked','Embarked_ind']]
print(df.head(10))


# # Target variable distribution

# In[20]:


c = df.Survived.value_counts(dropna=False)
p = df.Survived.value_counts(dropna=False, normalize=True)
pd.concat([c,p], axis=1, keys=['counts', '%']).to_excel("Target_Variable_Distribution.xlsx", header=True)
print(c)
print(round(p*100,1))


# # Splitting Continuous and Categorical types variables

# In[21]:


cols=df.columns
num_columns=df._get_numeric_data().columns
cat_columns=list(set(cols) - set(num_columns))
print("Total Columns : " + str(len(cols)))

df_categorical = df[cat_columns]
df_continuous = df[num_columns]
print(df_categorical.info())
print(df_continuous.info())


# # Variable Univariate Analysis

# # Continuous Field : Descriptive Statistics

# In[22]:


desc = df.describe()
df_univariate = desc.T
df_nmis = df.isnull().sum()
df_univariate = df_univariate.reset_index().set_index('index', drop=False)
df_nmis = df_nmis.reset_index().set_index('index', drop=False)
df_univariate_2 = pd.concat([df_nmis,df_univariate], axis=1, join_axes=[df_nmis.index], join = 'outer')
df_univariate_2 = df_univariate_2.rename(columns={ df_univariate_2.columns[1]: "nmiss" })
df_univariate_2 = df_univariate_2.drop([df_univariate_2.columns[0], df_univariate_2.columns[2]], axis=1)

nans = lambda df: df[df.isnull().any(axis=1)]

def descriptive_analysis(df_continuous,target_var,out_file):
    df_percentile = df_continuous.quantile([.01, .05, .1, .9, .95, .99]).T
    df_percentile = df_percentile.reset_index().set_index('index', drop=False)
    df_univariate_3 = pd.concat([df_univariate_2,df_percentile], axis=1, join_axes=[df_univariate_2.index], join = 'outer')
    df_univariate_3 = df_univariate_3.rename(columns={ df_univariate_3.columns[len(df_univariate_3.keys())-1]: "p99" , df_univariate_3.columns[len(df_univariate_3.keys())-2]: "p95", df_univariate_3.columns[len(df_univariate_3.keys())-3]: "p90", df_univariate_3.columns[len(df_univariate_3.keys())-4]: "p10", df_univariate_3.columns[len(df_univariate_3.keys())-5]: "p5", df_univariate_3.columns[len(df_univariate_3.keys())-6]: "p1"})
    df_univariate_3 = df_univariate_3.drop([df_univariate_3.columns[len(df_univariate_3.keys())-7]], axis=1)
    df_continuous_corr = pd.DataFrame(df_continuous.drop(target_var, axis=1).apply(lambda x: x.corr(df_continuous.eval(target_var))))
    df_continuous_corr = df_continuous_corr.reset_index().set_index('index', drop=False)
    df_continuous_corr.rename(columns={df_continuous_corr.columns[len(df_continuous_corr.keys())-1]: "corr"}, inplace=True)
    df_univariate_4 = pd.concat([df_univariate_3,df_continuous_corr], axis=1, join_axes=[df_univariate_3.index], join = 'outer')
    df_univariate_4 = df_univariate_4.drop([df_univariate_4.columns[len(df_univariate_4.keys())-2]], axis=1)

    df_continuous_column = list(df_continuous.columns)
    df_continuous_column = pd.Series(df_continuous_column)
    df_univariate_4 = df_univariate_4.reset_index()
    df_univariate_4 = df_univariate_4.rename(columns={'index':'variable','count':'n','50%':'median','25%':'p25','75%':'p75'})
    df_univariate_4 = df_univariate_4[['variable','n','nmiss','min','max','mean','median','p1','p5','p10','p25','p75','p90','p95','p99','corr']]
    df_univariate_4.sort_values(by=['variable'], ascending=False).dropna().to_excel(out_file, index = None, header=True)    
    print(df_univariate_4.sort_values(by=['variable'], ascending=False))
    
descriptive_analysis(df_continuous,target_var="Survived",out_file="Descriptive_stat_Continuous_feb24.xlsx")


# # Continuous Field : WOE and IV Calculation

# In[23]:

def create_volume_group(df_continuous,curr_var,target_col,n_bin):
    global df_fnl
    df_continuous_column=[]
    df_continuous_column.append(curr_var)
    ttl_vol=len(df_continuous)
    avg_bin_vol = round(ttl_vol/n_bin)
    lst=[]
    df_fnl=pd.DataFrame(lst)
    df_mod=pd.DataFrame(lst)
    df_mod_null=pd.DataFrame(lst)

    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):
            curr_var=df_continuous_column[i]

            df_mod1=pd.DataFrame(df_continuous[[curr_var,target_col]])

            #### Sort the Data ####
            df_mod1=df_mod1.sort_values(by=[curr_var],ascending=True)
            df_mod1=df_mod1.reset_index()
        
            df_mod_null=df_mod1[pd.isnull(df_mod1[curr_var])]
            df_mod1=df_mod1.dropna(subset=[curr_var]) 
        
            seq=list(range(1,len(df_mod1)+int(1)))
            df_seq=pd.DataFrame(seq,columns=['sequence'])

            df_mod1=pd.concat([df_mod1,df_seq],axis=1)
        
            if len(df_mod_null) > int(0):
                ttl_vol=len(df_mod1)
                avg_bin_vol = round(ttl_vol/n_bin)
                group_num='missing'
        
            pos_indx_max = 0
            val_list=df_mod1[curr_var].unique()
            
            if len(val_list) == 2:
                for i in range(len(val_list)):
                    val_of_indx=val_list[i]
                    df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                    group_num=i+int(1)
                    df_mod3['Decile']=group_num
                    df_fnl=pd.concat([df_fnl,df_mod3])
                    
            if len(val_list) != 2:        
                for bin_num in range(n_bin):
                    if  pos_indx_max < ttl_vol-1 :
                        if bin_num == 0:
                            indx = (bin_num+int(1)) * avg_bin_vol
                        else:
                            indx = int(pos_indx_max) + int(avg_bin_vol)
                
                        if indx > ttl_vol:
                            indx = ttl_vol-int(1)

                        val_of_indx = df_mod1[curr_var].iat[indx]
                                        
                        if math.isnan(val_of_indx) == True:
                            pos_indx_min=pos_indx_max+int(1)
                            pos_indx_max=ttl_vol-int(1)
                        else:
                            df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                            pos_indx_min = pos_indx_max
                            pos_indx_max = df_mod3['sequence'].iat[-1]
                       
                        df_mod3=df_mod1[pos_indx_min:pos_indx_max]
                        group_num=bin_num+int(1)
                        df_mod3['Decile']=group_num
                    
                        df_fnl=pd.concat([df_fnl,df_mod3])
            
            df_fnl=pd.concat([df_fnl,df_mod_null])
            
def cont_bin_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    global continuous_target_nx

    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].astype(object)
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].fillna('Missing')
    df_dcl_fnl['Decile'] = df_dcl_fnl['Decile'].fillna('Missing')
    

    continuous_target_nx = pd.DataFrame([df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
                                                 df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
                                                 df_dcl_fnl[(df_dcl_fnl[target_col] == 1)].groupby('Decile')[df_continuous_column[i]].count(),
                                                 df_dcl_fnl[(df_dcl_fnl[target_col] == 0)].groupby('Decile')[df_continuous_column[i]].count(),
                                                 df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]).T
    continuous_target_nx.columns = ["Min","Max","Event","Non-Event","Total"]

    continuous_target_nx.Event=continuous_target_nx.Event.fillna(0)
    continuous_target_nx['Non-Event']=continuous_target_nx['Non-Event'].fillna(0)
    
    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.Min)):
        list1.append(str(continuous_target_nx['Min'][i])+'-'+str(continuous_target_nx['Max'][i]))
        list_vol_pct.append(continuous_target_nx['Total'][i]/continuous_target_nx['Total'].sum())
        list_event_pct.append(continuous_target_nx['Event'][i]/continuous_target_nx['Total'][i])
    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx=continuous_target_nx.reindex(columns=["Decile","Min","Max",0,"Event","Non-Event","Total",1,2])
    continuous_target_nx = continuous_target_nx[["Decile","Min","Max",0,"Event","Non-Event","Total",1,2]]


    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-6]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-9]: "BINS"})       
    continuous_target_nx = continuous_target_nx.append({"Bins":"Total",
                                 "Min":" ",
                                 "Max":" ",
                                 "Range":" ",
                                 "Event":continuous_target_nx['Event'].sum(),
                                 "Non-Event":continuous_target_nx['Non-Event'].sum(),
                                 "Total":continuous_target_nx['Total'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Event(%)":continuous_target_nx['Event'].sum()/continuous_target_nx['Total'].sum()
                                                       },ignore_index=True)
    
def cont_bin_NO_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    global continuous_target_nx

    continuous_target_nx = pd.DataFrame([df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
                                         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
                                         df_dcl_fnl[(df_dcl_fnl[target_col] == 1)].groupby('Decile')[df_continuous_column[i]].count(),
                                         df_dcl_fnl[(df_dcl_fnl[target_col] == 0)].groupby('Decile')[df_continuous_column[i]].count(),
                                         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]).T
    continuous_target_nx.columns = ["Min","Max","Event","Non-Event","Total"]

    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.Min)):
        list1.append(str(continuous_target_nx['Min'][i])+'-'+str(continuous_target_nx['Max'][i]))
        list_vol_pct.append(continuous_target_nx['Total'][i]/continuous_target_nx['Total'].sum())
        list_event_pct.append(continuous_target_nx['Event'][i]/continuous_target_nx['Total'][i])
    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx = continuous_target_nx[["Decile","Min","Max",0,"Event","Non-Event","Total",1,2]]
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-6]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-9]: "BINS"})       
    continuous_target_nx = continuous_target_nx.append({"Bins":"Total",
                                 "Min":" ",
                                 "Max":" ",
                                 "Range":" ",
                                 "Event":continuous_target_nx['Event'].sum(),
                                 "Non-Event":continuous_target_nx['Non-Event'].sum(),
                                 "Total":continuous_target_nx['Total'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Event(%)":continuous_target_nx['Event'].sum()/continuous_target_nx['Total'].sum()
                                                       },ignore_index=True)
    

def calc_iv(var_name):
    global continuous_target_nx
    global data_fnl
    global IV_lst
    lst=[]
    
    continuous_target_nx.Event=continuous_target_nx.Event.fillna(0)
    continuous_target_nx['Non-Event']=continuous_target_nx['Non-Event'].fillna(0)
    row_cnt_without_total=len(continuous_target_nx)-int(1)
    
    for i in range(len(continuous_target_nx)):
        data_bin = continuous_target_nx.Bins[i]
        data_max = continuous_target_nx.Min[i]
        data_min = continuous_target_nx.Max[i]
        data_Value = continuous_target_nx.Range[i]
        data_All = int(continuous_target_nx.Total[i])
        data_Target_1 = int(continuous_target_nx.Event[i])
        data_Target_0 = int(continuous_target_nx['Non-Event'][i])
        data_Target_1_Rate = continuous_target_nx.Event[i] / continuous_target_nx.Total[i]
        data_Target_0_Rate = continuous_target_nx['Non-Event'][i] / continuous_target_nx.Total[i]
        data_Distribution_Target_1 = int(continuous_target_nx['Event'][i])/ continuous_target_nx['Event'].head(row_cnt_without_total).sum().sum()
        data_Distribution_Target_0 = int(continuous_target_nx['Non-Event'][i])/continuous_target_nx['Non-Event'].head(row_cnt_without_total).sum()
        #'WOE' value by ln(Distribution Good/Distribution Bad)
        data_WoE = np.log(data_Distribution_Target_1 / data_Distribution_Target_0)
        
        if (data_WoE == np.inf) or (data_WoE == -np.inf):
            data_WoE = 0
            
        data_IV = data_WoE * (data_Distribution_Target_1 - data_Distribution_Target_0)
        data=[data_bin,data_min,data_max,data_Value,data_All,data_Target_1,data_Target_0,data_Target_1_Rate,data_Target_0_Rate,data_Distribution_Target_1,data_Distribution_Target_0,data_WoE,data_IV]
        lst.append(data)
    
    data_fnl = pd.DataFrame(lst,columns=['Bins', 'Min' , 'Max' , 'Range', 'All', 'Target_1', 'Target_0','Target_1_Rate','Target_0_Rate','Distribution_Target_1','Distribution_Target_0','WOE','IV'])
    iv_val=[var_name,data_fnl['IV'].head(row_cnt_without_total).sum()]
    IV_lst.append(iv_val)
    
def cont_bin(df,df_continuous,n_bin,target_col,filename):
    global continuous_target_nx
    global data_fnl
    global IV_lst
    global df_fnl
    
    df_continuous_column = list(df_continuous.columns)
    
    IV_lst=[]
    
    writer1 = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer1.book
    worksheet=workbook.add_worksheet('WOE')
    writer1.sheets['WOE'] = worksheet

    n = 0
    m = -1
    
    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):  
            print(df_continuous_column[i])
            create_volume_group(df_continuous,df_continuous_column[i],target_col,n_bin)
            df_fnl=df_fnl[[df_continuous_column[i],target_col,'Decile']]

            if df_fnl.eval(df_continuous_column[i]).isnull().sum() > 0:
                cont_bin_Miss(df_fnl,df_continuous_column,i,target_col,filename)                
            else:
                cont_bin_NO_Miss(df_fnl,df_continuous_column,i,target_col,filename)
            
            calc_iv(df_continuous_column[i])

            worksheet.write_string(n, 0, df_continuous_column[i])
            data_fnl.to_excel(writer1,sheet_name='WOE',startrow=n+1 , startcol=0,index = False)
            n += len(continuous_target_nx.index) + 4
    
    data_IV = pd.DataFrame(IV_lst,columns=['Variable','IV_value'])
    print(data_IV)

    data_IV.to_excel(writer1,sheet_name='IV',startrow=m+1 , startcol=0,index = False)       
    writer1.save()
cont_bin(df,df_continuous,n_bin=10,target_col='Survived',filename='Descriptive_Statistics_Continuous_WOE_10_feb24.xlsx')

# df : Total Dataframe
# df_continuous : Dataframe having only Continuous variables
# n_bin : Number of Bins 
# filename : Output File Name


# # Continuous Field : Rank and Plot

# In[25]:
# Creating two new directory in the working directory to store the Plots of variables 
work_dir=os.getcwd()
print(work_dir)
new_dir=work_dir+"/line_plot/"

check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))
    
    
new_dir=work_dir+"/scatter_plot/"
check_dir_present=os.path.isdir(new_dir)
if check_dir_present == False:
    os.mkdir(new_dir)
    print("New Directory created : " + str(new_dir))
else:
    print("Existing Directory used : " + str(new_dir))

def plot_stat(df_continuous,title_name,target_col):
    global df_plot,file_i,img_name,work_dir,scatter_img_name
    
    width=0.35
    df_plot['Volume(%)']=df_plot['Volume(%)']*100
    df_plot['Event(%)']=df_plot['Event(%)']*100

    df_plot.plot(x='Bins',y='Volume(%)',kind='bar',width=width,label=('Volume(%)'),color='b')
    plt.ylabel('Volume(%)',color='b')
    plt.ylim((0,100))
    
    plt.legend(loc='upper right')

    df_plot['Event(%)'].plot(secondary_y=True,label=('Event(%)'),color='r')
    plt.ylabel('Event(%)',color='r')
    plt.ylim((0,100))
    plt.legend(loc='upper right')
    
    axis_1=plt.gca()

    for i,j in zip(df_plot['Volume(%)'].index,df_plot['Volume(%)']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='k')
        axis_1.annotate('%s' %i,xy=(i,j),color='k')
        
    for i,j in zip(df_plot['Event(%)'].index,df_plot['Event(%)']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='k')
        axis_1.annotate('%s' %i,xy=(i,j),color='k')

    plt.xlim([-width,len(df_plot['Volume(%)'])-width])
    plt.title(title_name)
    plt.xlabel("Bins")
    plt.grid()
    img_name=str(work_dir)+str("/line_plot/")+str(title_name)+ str(".png")
    plt.savefig(img_name,dpi=300,bbox_inches='tight')
    plt.clf()
    
    fig=plt.figure(figsize=(6,4))
    plt.scatter(df_continuous[title_name],df_continuous[target_col],c='DarkBlue')
    plt.ylabel(target_col,color='b')
    plt.xlabel(title_name,color='b')
    plt.title(title_name)
    plt.grid()
    scatter_img_name=str(work_dir)+str("/scatter_plot/")+str(title_name)+ str(".png")
    plt.show()
    fig.savefig(scatter_img_name,dpi=300,bbox_inches='tight')
    plt.clf()

def add_table_plot(df_continuous,in_file,sheet_nm,out_file,target_col):
    
    global df_plot,img_name,work_dir,scatter_img_name
    
    work_dir=os.getcwd()
    df_cont=pd.read_excel(in_file,header=None,sheet_name=sheet_nm)
    df_cont.columns=['Bins','Min','Max','Range','Event','Non-Event','Total','Volume(%)','Event(%)']
    df_cont=df_cont.fillna('')
    wb=xlsxwriter.Workbook(out_file)
    ws=wb.add_worksheet(sheet_nm)
    
    for i in range(len(df_cont)):
        
        for j in range(len(df_cont.columns)):
            if j == 0:
                col_pos= 'A'
            if j == 1:
                col_pos= 'B'
            if j == 2:
                col_pos= 'C'  
            if j == 3:
                col_pos= 'D'
            if j == 4:
                col_pos= 'E'
            if j == 5:
                col_pos= 'F'
            if j == 6:
                col_pos= 'G'  
            if j == 7:
                col_pos= 'H'
            if j == 8:
                col_pos= 'I'

            wrt_pos=str(col_pos)+str(i)
            ws.write(wrt_pos,df_cont.iloc[i,j])

        if df_cont.iloc[:,0][i] == "Bins":
            pos_min_loc=i+int(1)
            if pos_min_loc==1:
                title_name=df_cont.columns[0]
            else:
                title_loc=int(pos_min_loc)-int(2)
                title_name=df_cont.iloc[:,0][title_loc]
        
        if df_cont.iloc[:,0][i] == "Total":
            pos_max_loc=i
            df_plot=df_cont[pos_min_loc:pos_max_loc]
            df_plot.columns=['Bins','Min','Max','Range','Event','Non-Event','Total','Volume(%)','Event(%)']
            df_plot=df_plot.reset_index()
            plot_stat(df_continuous,title_name,target_col)
            img_pos=str('K')+ str(pos_min_loc-int(2))   
            img2=mimg.imread(img_name)
            imgplot2=plt.imshow(img2)

            ws.insert_image(img_pos,img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
            
            scatter_img_pos=str('Q')+ str(pos_min_loc-int(2))  
            ws.insert_image(scatter_img_pos,scatter_img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
    wb.close()

def create_volume_group(df_continuous,curr_var,target_col,n_bin):
    global df_fnl
    df_continuous_column=[]
    df_continuous_column.append(curr_var)
    ttl_vol=len(df_continuous)
    avg_bin_vol = round(ttl_vol/n_bin)
    lst=[]
    df_fnl=pd.DataFrame(lst)
    df_mod=pd.DataFrame(lst)
    df_mod_null=pd.DataFrame(lst)

    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):
            curr_var=df_continuous_column[i]

            df_mod1=pd.DataFrame(df_continuous[[curr_var,target_col]])

            #### Sort the Data ####
            df_mod1=df_mod1.sort_values(by=[curr_var],ascending=True)
            df_mod1=df_mod1.reset_index()
        
            df_mod_null=df_mod1[pd.isnull(df_mod1[curr_var])]
            df_mod1=df_mod1.dropna(subset=[curr_var]) 
        
            seq=list(range(1,len(df_mod1)+int(1)))
            df_seq=pd.DataFrame(seq,columns=['sequence'])

            df_mod1=pd.concat([df_mod1,df_seq],axis=1)
        
            if len(df_mod_null) > int(0):
                ttl_vol=len(df_mod1)
                avg_bin_vol = round(ttl_vol/n_bin)
                group_num='missing'
        
            pos_indx_max = 0
            val_list=df_mod1[curr_var].unique()
            
            if len(val_list) == 2:
                for i in range(len(val_list)):
                    val_of_indx=val_list[i]
                    df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                    group_num=i+int(1)
                    df_mod3['Decile']=group_num
                    df_fnl=pd.concat([df_fnl,df_mod3])
                    
            if len(val_list) != 2:        
                for bin_num in range(n_bin):
                    if  pos_indx_max < ttl_vol-1 :
                        if bin_num == 0:
                            indx = (bin_num+int(1)) * avg_bin_vol
                        else:
                            indx = int(pos_indx_max) + int(avg_bin_vol)
                
                        if indx > ttl_vol:
                            indx = ttl_vol-int(1)

                        val_of_indx = df_mod1[curr_var].iat[indx]
                
                        if math.isnan(val_of_indx) == True:
                            pos_indx_min=pos_indx_max+int(1)
                            pos_indx_max=ttl_vol-int(1)
                        else:
                            df_mod3=df_mod1[df_mod1[curr_var] == val_of_indx]
                            pos_indx_min = pos_indx_max
                            pos_indx_max = df_mod3['sequence'].iat[-1]
                       
                        df_mod3=df_mod1[pos_indx_min:pos_indx_max]
                        group_num=bin_num+int(1)
                        df_mod3['Decile']=group_num
                    
                        df_fnl=pd.concat([df_fnl,df_mod3])
            
            df_fnl=pd.concat([df_fnl,df_mod_null])
            

def cont_bin_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    global continuous_target_nx

    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].astype(object)
    df_dcl_fnl[df_continuous_column[i]] = df_dcl_fnl[df_continuous_column[i]].fillna('Missing')
    df_dcl_fnl['Decile'] = df_dcl_fnl['Decile'].fillna('Missing')
    
    continuous_target_nx = pd.DataFrame([df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
                                                 df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
                                                 df_dcl_fnl[(df_dcl_fnl[target_col] == 1)].groupby('Decile')[df_continuous_column[i]].count(),
                                                 df_dcl_fnl[(df_dcl_fnl[target_col] == 0)].groupby('Decile')[df_continuous_column[i]].count(),
                                                 df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]).T
    continuous_target_nx.columns = ["Min","Max","Event","Non-Event","Total"]

    continuous_target_nx.Event=continuous_target_nx.Event.fillna(0)
    continuous_target_nx['Non-Event']=continuous_target_nx['Non-Event'].fillna(0)
    
    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.Min)):
        list1.append(str(continuous_target_nx['Min'][i])+'-'+str(continuous_target_nx['Max'][i]))
        list_vol_pct.append(continuous_target_nx['Total'][i]/continuous_target_nx['Total'].sum())
        list_event_pct.append(continuous_target_nx['Event'][i]/continuous_target_nx['Total'][i])
    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx=continuous_target_nx.reindex(columns=["Decile","Min","Max",0,"Event","Non-Event","Total",1,2])
    continuous_target_nx = continuous_target_nx[["Decile","Min","Max",0,"Event","Non-Event","Total",1,2]]

    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-6]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-9]: "Bins"})       
    continuous_target_nx = continuous_target_nx.append({"Bins":"Total",
                                 "Min":" ",
                                 "Max":" ",
                                 "Range":" ",
                                 "Event":continuous_target_nx['Event'].sum(),
                                 "Non-Event":continuous_target_nx['Non-Event'].sum(),
                                 "Total":continuous_target_nx['Total'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Event(%)":continuous_target_nx['Event'].sum()/continuous_target_nx['Total'].sum()
                                                       },ignore_index=True)
    
def cont_bin_NO_Miss(df_dcl_fnl,df_continuous_column,i,target_col,filename):
    global continuous_target_nx

    continuous_target_nx = pd.DataFrame([df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].min(),
                                         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].max(),
                                         df_dcl_fnl[(df_dcl_fnl[target_col] == 1)].groupby('Decile')[df_continuous_column[i]].count(),
                                         df_dcl_fnl[(df_dcl_fnl[target_col] == 0)].groupby('Decile')[df_continuous_column[i]].count(),
                                         df_dcl_fnl.groupby('Decile')[df_continuous_column[i]].count()]).T
    continuous_target_nx.columns = ["Min","Max","Event","Non-Event","Total"]

    continuous_target_nx=continuous_target_nx.reset_index()
    list1=[]
    list_vol_pct=[]
    list_event_pct=[]
    for i in range(len(continuous_target_nx.Min)):
        list1.append(str(continuous_target_nx['Min'][i])+'-'+str(continuous_target_nx['Max'][i]))
        list_vol_pct.append(continuous_target_nx['Total'][i]/continuous_target_nx['Total'].sum())
        list_event_pct.append(continuous_target_nx['Event'][i]/continuous_target_nx['Total'][i])
    continuous_target_nx = pd.concat([pd.Series(list1),continuous_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    continuous_target_nx = continuous_target_nx[["Decile","Min","Max",0,"Event","Non-Event","Total",1,2]]
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-6]: "Range"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-2]: "Volume(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-1]: "Event(%)"})
    continuous_target_nx = continuous_target_nx.rename(columns={continuous_target_nx.columns[len(continuous_target_nx.keys())-9]: "Bins"})       
    continuous_target_nx = continuous_target_nx.append({"Bins":"Total",
                                 "Min":" ",
                                 "Max":" ",
                                 "Range":" ",
                                 "Event":continuous_target_nx['Event'].sum(),
                                 "Non-Event":continuous_target_nx['Non-Event'].sum(),
                                 "Total":continuous_target_nx['Total'].sum(),
                                 "Volume(%)":continuous_target_nx['Volume(%)'].sum(),
                                 "Event(%)":continuous_target_nx['Event'].sum()/continuous_target_nx['Total'].sum()
                                                       },ignore_index=True)
    
    
def cont_bin(df,df_continuous,n_bin,target_col,filename):
    global continuous_target_nx,df_fnl
    
    df_continuous_column = list(df_continuous.columns)
    writer = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer.book
    worksheet=workbook.add_worksheet('Continous')
    writer.sheets['Continous'] = worksheet
    n = 1
    
    for i in range(len(df_continuous_column)):
        if (df_continuous_column[i] != target_col):        
            create_volume_group(df_continuous,df_continuous_column[i],target_col,n_bin)
            df_fnl=df_fnl[[df_continuous_column[i],target_col,'Decile']]
            
            if df_fnl.eval(df_continuous_column[i]).isnull().sum() > 0:
                cont_bin_Miss(df_fnl,df_continuous_column,i,target_col,filename)                
            else:
                cont_bin_NO_Miss(df_fnl,df_continuous_column,i,target_col,filename)
                
            worksheet.write_string(n, 0, df_continuous_column[i])
            continuous_target_nx.to_excel(writer,sheet_name='Continous',startrow=n+1 , startcol=0,index = False)
            n += len(continuous_target_nx.index) + 10
           
    writer.save()
        
cont_bin(df,df_continuous,n_bin=10,target_col='Survived',filename='Continuous_base_feb24.xlsx')        
add_table_plot(df_continuous,in_file='Continuous_base_feb24.xlsx',sheet_nm='Continous',target_col='Survived',out_file='Continuous_rank_plot_feb24.xlsx')    
 
# Note : The File Created from cont_bin() should be used as input filename in add_table_plot() to add the Plot of individual features
# df : Total Dataframe
# df_continuous : Dataframe having only Continuous variables
# n_bin : Number of Bins 
# filename : Output File Name


# # Variables Trend - Categorical

# # Categorical fields : WOE and IV calculation

# In[26]:


def calc_iv(var_name):
    global categorical_target_nx
    global data_fnl
    global IV_lst
    lst=[]
    
    categorical_target_nx.Event=categorical_target_nx.Event.fillna(0)
    categorical_target_nx['Non-Event']=categorical_target_nx['Non-Event'].fillna(0)
    row_cnt_without_total=len(categorical_target_nx)-int(1)
    
    for i in range(len(categorical_target_nx)):
        data_bin = categorical_target_nx.Levels[i]
        data_All = int(categorical_target_nx.Total[i])
        data_Target_1 = int(categorical_target_nx.Event[i])
        data_Target_0 = int(categorical_target_nx['Non-Event'][i])
        data_Target_1_Rate = categorical_target_nx.Event[i] / categorical_target_nx.Total[i]
        data_Target_0_Rate = categorical_target_nx['Non-Event'][i] / categorical_target_nx.Total[i]
        data_Distribution_Target_1 = int(categorical_target_nx['Event'][i])/ categorical_target_nx['Event'].head(row_cnt_without_total).sum().sum()
        data_Distribution_Target_0 = int(categorical_target_nx['Non-Event'][i])/categorical_target_nx['Non-Event'].head(row_cnt_without_total).sum()
        #'WOE' value by ln(Distribution Good/Distribution Bad)
        data_WoE = np.log(data_Distribution_Target_1 / data_Distribution_Target_0)
        #data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
        #data_WoE=data_WoE.replace(np.inf,0).replace(-np.inf,0)
        
        if (data_WoE == np.inf) or (data_WoE == -np.inf):
            data_WoE = 0
            
        data_IV = data_WoE * (data_Distribution_Target_1 - data_Distribution_Target_0)
        data=[data_bin,data_All,data_Target_1,data_Target_0,data_Target_1_Rate,data_Target_0_Rate,data_Distribution_Target_1,data_Distribution_Target_0,data_WoE,data_IV]
        lst.append(data)
    
    data_fnl = pd.DataFrame(lst,columns=['Levels', 'Total', 'Event', 'Non-Event','Target_1_Rate','Target_0_Rate','Distribution_Target_1','Distribution_Target_0','WOE','IV'])
    iv_val=[var_name,data_fnl['IV'].head(row_cnt_without_total).sum()]
    IV_lst.append(iv_val)

    
def cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename):

    global categorical_target_nx

    categorical_target_nx = pd.DataFrame([df_cat_fnl[(df_cat_fnl[target_col] == 1)].groupby('Levels')[df_categorical_column[i]].count(),
                                         df_cat_fnl[(df_cat_fnl[target_col] == 0)].groupby('Levels')[df_categorical_column[i]].count(),
                                         df_cat_fnl.groupby('Levels')[df_categorical_column[i]].count()]).T

    categorical_target_nx.columns = ["Event","Non-Event","Total"]    
    categorical_target_nx['Event'] = categorical_target_nx['Event'].fillna(0)
    categorical_target_nx['Non-Event'] = categorical_target_nx['Non-Event'].fillna(0)
    categorical_target_nx['Total'] = categorical_target_nx['Total'].fillna(0)

    categorical_target_nx=categorical_target_nx.reset_index()
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[0]: "Levels"})    
        
    list_vol_pct=[]
    list_event_pct=[]

    for j in range(len(categorical_target_nx.Event)):

        list_vol_pct.append(categorical_target_nx['Total'][j]/categorical_target_nx['Total'].sum())
        list_event_pct.append(categorical_target_nx['Event'][j]/categorical_target_nx['Total'][j])
    
    categorical_target_nx = pd.concat([categorical_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    
    
    categorical_target_nx = categorical_target_nx[["Levels","Event","Non-Event","Total",0,1]]        
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-2]: "Volume(%)"})
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-1]: "Event(%)"})
    categorical_target_nx = categorical_target_nx.sort_values(by=['Total'], ascending=False)
    
    
    categorical_target_nx = categorical_target_nx.append({"Levels":"Total",
                                     "Event":categorical_target_nx['Event'].sum(),
                                     "Non-Event":categorical_target_nx['Non-Event'].sum(),
                                     "Total":categorical_target_nx['Total'].sum(),
                                     "Volume(%)":categorical_target_nx['Volume(%)'].sum(),                                                              "Event(%)":categorical_target_nx['Event'].sum()/categorical_target_nx['Total'].sum()
                                                           },ignore_index=True)
                 

def cat_bin(df,df_categorical,target_col,filename):

    global categorical_target_nx,data_fnl,IV_lst
    
    df_categorical_column = list(df_categorical.columns)
    
    IV_lst=[]
    
    writer1 = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer1.book
    worksheet=workbook.add_worksheet('WOE')
    writer1.sheets['WOE'] = worksheet

    n = 0
    m = -1   
    for i in range(len(df_categorical_column)):

        if (df_categorical_column[i] != target_col):       

            nparray_cat=df[df_categorical_column[i]].fillna('Missing').unique()
            nparray_sort=np.sort(nparray_cat)        
            df_cat = pd.concat([pd.Series(nparray_sort),pd.Series(nparray_sort)],axis=1, keys=[df_categorical_column[i],'Levels'])        
            df_tst = df.loc[:, [df_categorical_column[i],target_col]].sort_values(by=[df_categorical_column[i]]).fillna('Missing')
            df_cat_fnl = pd.merge(df_tst, df_cat, how='left', on=[df_categorical_column[i]])                    
            cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename)                                   
            calc_iv(df_categorical_column[i])
     
            worksheet.write_string(n, 0, df_categorical_column[i])
            data_fnl.to_excel(writer1,sheet_name='WOE',startrow=n+1 , startcol=0,index = False)
            n += len(categorical_target_nx.index) + 4
    
    data_IV = pd.DataFrame(IV_lst,columns=['Variable','IV_value'])
    data_IV.to_excel(writer1,sheet_name='IV',startrow=m+1 , startcol=0,index = False)       
    writer1.save()

cat_bin(df,df_categorical,target_col='Survived',filename='Descriptive_Statistics_Categorical_WOE_feb24.xlsx') 

# df : Total Dataframe
# df_categorical : Dataframe having only Categorical variables
# target_col : Target Column name 
# filename : Output File Name


# # Categorical Field : Rank and Plot

def plot_stat(df_categorical,title_name,target_col):
    global df_plot,file_i,img_name,work_dir,scatter_img_name
    
    width=0.35
    df_plot['Volume(%)']=df_plot['Volume(%)']*100
    df_plot['Event(%)']=df_plot['Event(%)']*100

    df_plot.plot(x='Levels',y='Volume(%)',kind='bar',width=width,label=('Volume(%)'),color='b')
    y_pos=range(len(df_plot['Levels']))
    plt.ylabel('Volume(%)',color='b')
    plt.ylim((0,100))
    plt.legend(loc='upper right')

    df_plot['Event(%)'].plot(secondary_y=True,label=('Event(%)'),color='r',rot=90)
    plt.ylabel('Event(%)',color='r')
    plt.ylim((0,100))
    plt.legend(loc='upper right')
    
    axis_1=plt.gca()

    for i,j in zip(df_plot['Volume(%)'].index,df_plot['Volume(%)']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='k')
        axis_1.annotate('%s' %i,xy=(i,j),color='k')
        
    for i,j in zip(df_plot['Event(%)'].index,df_plot['Event(%)']):
        i=round(i,2)
        j=round(j,2)
        axis_1.annotate('%s' %j,xy=(i,j),color='k')
        axis_1.annotate('%s' %i,xy=(i,j),color='k')

    plt.xlim([-width,len(df_plot['Volume(%)'])-width])
    plt.title(title_name)
    plt.xlabel('Levels')
    plt.grid()
    img_name=str(work_dir)+str("/line_plot/")+str(title_name)+ str(".png")
    plt.savefig(img_name,dpi=300,bbox_inches='tight')
    plt.clf()
    
    fig=plt.figure(figsize=(6,4))
    df_categorical[title_name]=df_categorical[title_name].fillna("Missing")
    plt.scatter(df_categorical[title_name],df_categorical[target_col],c='DarkBlue')
    plt.ylabel(target_col,color='b')
    plt.xlabel(title_name,color='b')
    plt.xticks(rotation=90)
    plt.title(title_name)
    plt.grid()
    scatter_img_name=str(work_dir)+str("/scatter_plot/")+str(title_name)+ str(".png")
    plt.show()
    fig.savefig(scatter_img_name,dpi=300,bbox_inches='tight')
    plt.clf()

def add_table_plot(df_categorical,in_file,sheet_nm,out_file,target_col):
    
    global df_plot,img_name,work_dir,scatter_img_name
    
    work_dir=os.getcwd()
    df_cont=pd.read_excel(in_file,header=None,sheet_name=sheet_nm)
    df_cont.columns=['Levels','Event','Non-Event','Total','Volume(%)','Event(%)']
    df_cont=df_cont.fillna('')
    wb=xlsxwriter.Workbook(out_file)
    ws=wb.add_worksheet(sheet_nm)
    
    for i in range(len(df_cont)):
        
        for j in range(len(df_cont.columns)):
            if j == 0:
                col_pos= 'A'
            if j == 1:
                col_pos= 'B'
            if j == 2:
                col_pos= 'C'  
            if j == 3:
                col_pos= 'D'
            if j == 4:
                col_pos= 'E'
            if j == 5:
                col_pos= 'F'
            if j == 6:
                col_pos= 'G'  
            if j == 7:
                col_pos= 'H'
            if j == 8:
                col_pos= 'I'

            wrt_pos=str(col_pos)+str(i)
            ws.write(wrt_pos,df_cont.iloc[i,j])
            
        if df_cont.iloc[:,0][i] == "Levels":
            pos_min_loc=i+int(1)
            if pos_min_loc==1:
                title_name=df_cont.columns[0]
            else:
                title_loc=int(pos_min_loc)-int(2)
                title_name=df_cont.iloc[:,0][title_loc]
        
        if df_cont.iloc[:,0][i] == "Total":
            pos_max_loc=i
            df_plot=df_cont[pos_min_loc:pos_max_loc]
            df_plot.columns=['Levels','Event','Non-Event','Total','Volume(%)','Event(%)']
            df_plot=df_plot.reset_index()
            plot_stat(df_categorical,title_name,target_col)
            img_pos=str('H')+ str(pos_min_loc-int(2))   
            img2=mimg.imread(img_name)
            imgplot2=plt.imshow(img2)

            ws.insert_image(img_pos,img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})

            scatter_img_pos=str('N')+ str(pos_min_loc-int(2))  
            ws.insert_image(scatter_img_pos,scatter_img_name,{'x_scale' : 0.6, 'y_scale' : 0.6})
    wb.close()

    
def cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename):

    global categorical_target_nx

    categorical_target_nx = pd.DataFrame([df_cat_fnl[(df_cat_fnl[target_col] == 1)].groupby('Levels')[df_categorical_column[i]].count(),
                                         df_cat_fnl[(df_cat_fnl[target_col] == 0)].groupby('Levels')[df_categorical_column[i]].count(),
                                         df_cat_fnl.groupby('Levels')[df_categorical_column[i]].count()]).T

    categorical_target_nx.columns = ["Event","Non-Event","Total"] 
    categorical_target_nx['Event'] = categorical_target_nx['Event'].fillna(0)
    categorical_target_nx['Non-Event'] = categorical_target_nx['Non-Event'].fillna(0)
    categorical_target_nx['Total'] = categorical_target_nx['Total'].fillna(0)

    categorical_target_nx=categorical_target_nx.reset_index()
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[0]: "Levels"})    
        
    list_vol_pct=[]
    list_event_pct=[]

    for j in range(len(categorical_target_nx.Event)):

        list_vol_pct.append(categorical_target_nx['Total'][j]/categorical_target_nx['Total'].sum())
        list_event_pct.append(categorical_target_nx['Event'][j]/categorical_target_nx['Total'][j])
    
    categorical_target_nx = pd.concat([categorical_target_nx,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)
    
    categorical_target_nx = categorical_target_nx[["Levels","Event","Non-Event","Total",0,1]]        
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-2]: "Volume(%)"})
    categorical_target_nx = categorical_target_nx.rename(columns={categorical_target_nx.columns[len(categorical_target_nx.keys())-1]: "Event(%)"})
    categorical_target_nx = categorical_target_nx.sort_values(by=['Total'], ascending=False)
    

    categorical_target_nx = categorical_target_nx.append({"Levels":"Total",
                                     "Event":categorical_target_nx['Event'].sum(),
                                     "Non-Event":categorical_target_nx['Non-Event'].sum(),
                                     "Total":categorical_target_nx['Total'].sum(),
                                     "Volume(%)":categorical_target_nx['Volume(%)'].sum(),                                     "Event(%)":categorical_target_nx['Event'].sum()/categorical_target_nx['Total'].sum()
                                                           },ignore_index=True)
                 

def cat_bin(df,df_categorical,target_col,filename):

    global categorical_target_nx

    df_categorical_column = list(df_categorical.columns)

    writer = pd.ExcelWriter(filename,engine='xlsxwriter')
    workbook=writer.book
    worksheet=workbook.add_worksheet('Categorical')
    writer.sheets['Categorical'] = worksheet

    n = 1    
    for i in range(len(df_categorical_column)):

        if (df_categorical_column[i] != target_col):       
            nparray_cat=df[df_categorical_column[i]].fillna('Missing').unique()
            nparray_sort=np.sort(nparray_cat)        
            df_cat = pd.concat([pd.Series(nparray_sort),pd.Series(nparray_sort)],axis=1, keys=[df_categorical_column[i],'Levels'])        
            df_tst = df.loc[:, [df_categorical_column[i],target_col]].sort_values(by=[df_categorical_column[i]]).fillna('Missing')
            df_cat_fnl = pd.merge(df_tst, df_cat, how='left', on=[df_categorical_column[i]])                    
            cat_bin_trend(df_cat_fnl,df_categorical_column,i,target_col,filename)                                   

            worksheet.write_string(n, 0, df_categorical_column[i])
            categorical_target_nx.to_excel(writer,sheet_name='Categorical',startrow=n+1 , startcol=0,index = False)
            n += len(categorical_target_nx.index) + 10

    writer.save()      

cat_bin(df,df_categorical,target_col='Survived',filename='Categorical_base_feb24.xlsx')       
add_table_plot(df,in_file='Categorical_base_feb24.xlsx',sheet_nm='Categorical',target_col='Survived',out_file='Categorical_rank_plot_feb24.xlsx')

# Note : The File Created from cat_bin() should be used as input filename in add_table_plot() to add the Plot of individual features
# df : Total Dataframe
# df_categorical : Dataframe having only Categorical variables
# filename : Output File Name

# # Replace with Weight of Evidence for categorical Fields 
# Provide the Output file created in the previous step(Categorical fields : WOE and IV calculation) as input to the below line
woe_df=pd.read_excel('Descriptive_Statistics_Categorical_WOE_feb24.xlsx',sheet_name='WOE',header=None)

for cat_i in list(df_categorical.columns):
    match_fnd=""
    new_col=str(cat_i)+ "_WOE"
    
    for j in range(len(woe_df)):        
        if str(cat_i) == str(woe_df.iloc[j][0]) and match_fnd =="":            
            match_fnd='y'
            
        if str(woe_df.iloc[j][0]) == "Total":
            match_fnd = ""
            
        if match_fnd == 'y':            
            if (str(woe_df.iloc[j][0]) != "Levels") :
                #print(woe_df.iloc[j])
                if (str(cat_i) == str(woe_df.iloc[j][0])):
                    woe_ln=str("df['") + str(cat_i) + str("_WOE']=0")
                else:
                    
                    if  str(woe_df.iloc[j][0]) == "Missing_need_replace":
                        woe_ln = str("df.loc[df['") + str(cat_i) + str("'].isna()")  + str(",'") + str(cat_i) + str("_WOE']=") + str(woe_df.iloc[j][8])
                        
                        df.loc[df[cat_i].isna(),new_col]=woe_df.iloc[j][8]

                    else:
                        woe_ln = str("df.loc[df['") + str(cat_i) + str("']==") + str('"') + str(woe_df.iloc[j][0]) + str('"') + str(",'") + str(cat_i) + str("_WOE']=") + str(woe_df.iloc[j][8])
                        
                        df.loc[df[cat_i]==str(woe_df.iloc[j][0]),new_col]=woe_df.iloc[j][8]
                        
                                           
                
                print(woe_ln)

            

df.columns=df.keys()
df.info(verbose = True)
pd.DataFrame(df.dtypes).reset_index().rename(columns={'index':'variable' , 0:'Data Types'}).to_excel("Variable_List1.xlsx", header=True)
# # List of Variables for Missing Treatment
# Total
select_var =['Survived','Pclass','Sex','Age','Child_ind','SibSp','Parch','Dependent_ind','Parent_child_ind','Sibling_Spouse_ind','Fare','Cabin','Embarked','Embarked_ind','Dependent_ind_WOE','Cabin_WOE','Embarked_ind_WOE','Embarked_WOE','Parent_child_ind_WOE','Sibling_Spouse_ind_WOE','Sex_WOE','Child_ind_WOE']
select_cont=['Pclass','Age','SibSp','Parch','Fare']

X_cont=df[select_cont]

X=df[select_var]
y=df['Survived']  # Labels
# # Missing Field Identification
df_missing=0
def missing_values_table(df_phase3):
    mis_val = df_phase3.isnull().sum()
    mis_val_percent = 100*df_phase3.isnull().sum()/len(df_phase3)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 'columns')
    mis_val_table_ren_columns = mis_val_table.rename(columns ={0: 'Count of Missing Values', 1: '% of Total Values'} )

    #Sort the table by percent of missing values descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0 ].sort_values(by='% of Total Values', ascending = False).round(1)
    mis_val_table_ren_columns = mis_val_table_ren_columns.reset_index()
    #print summary
    #print("The dataset has " + str(df_phase3.shape[0]) + " rows and " + str(df_phase3.shape[1]) + " columns." )
    #print(str(mis_val_table_ren_columns.shape[0]) )

    return mis_val_table_ren_columns;

df_missing=missing_values_table(X)
df_missing_cont=missing_values_table(X_cont)
print(df_missing)


# # Missing Field Treatment with Mean
Age_mean=df['Age'].mean()
df['Age']=df['Age'].fillna(Age_mean)

# # variance_inflation_factor


select_var =['Age','Cabin_WOE','Child_ind_WOE','Dependent_ind_WOE','Embarked_ind_WOE','Fare','Parch','Parent_child_ind_WOE','Pclass','Sex_WOE','Sibling_Spouse_ind_WOE','SibSp']
X=df[select_var]
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif


# # Correlation matrix
select_var =['Age','Cabin_WOE','Child_ind_WOE','Dependent_ind_WOE','Embarked_ind_WOE','Fare','Parch','Parent_child_ind_WOE','Pclass','Sex_WOE','Sibling_Spouse_ind_WOE','SibSp']
corr_df=df[select_var]
corr=corr_df.corr()
corr.to_excel("titanic.correlation_matrix.xlsx")
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# # Train Test Split
select_var =['Survived','Pclass','Sex','Age','Child_ind','SibSp','Parch','Dependent_ind','Parent_child_ind','Sibling_Spouse_ind','Fare','Cabin','Embarked','Embarked_ind','Dependent_ind_WOE','Cabin_WOE','Embarked_ind_WOE','Embarked_WOE','Parent_child_ind_WOE','Sibling_Spouse_ind_WOE','Sex_WOE','Child_ind_WOE']

X=df[select_var]
y=df['Survived']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)  

# # Decision Tree Model

def decision_tree_classification():
    global X_train,y_train,X_test,y_test
    global decision_tree_model
    from sklearn.tree import DecisionTreeClassifier
    
    decision_tree_model = DecisionTreeClassifier(max_depth=8,random_state=100)
    decision_tree_model.fit(X_train,y_train)
    y_pred =decision_tree_model.predict(X_train)

    print("====== Decision_Tree_Classification Train ======")
    print(" Accuracy : "  + str(metrics.accuracy_score(y_train,y_pred)))
    print(" Recall : "  + str(metrics.recall_score(y_train,y_pred)))
    print(" Precision : "  + str(metrics.precision_score(y_train,y_pred)))
    print(" F1_Score : "  + str(metrics.f1_score(y_train,y_pred)))
    print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_train,y_pred)))
    print(" ")

    y_pred = decision_tree_model.predict(X_test)
    
    print("====== Decision_Tree_Classification Test ======")
    print(" Accuracy : "  + str(metrics.accuracy_score(y_test,y_pred)))
    print(" Recall : "  + str(metrics.recall_score(y_test,y_pred)))
    print(" Precision : "  + str(metrics.precision_score(y_test,y_pred)))
    print(" F1_Score : "  + str(metrics.f1_score(y_test,y_pred)))
    print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_test,y_pred)))
    print(" ")
         
select_var=['Pclass','Child_ind','Fare','Sex_WOE','Dependent_ind_WOE','Embarked_ind_WOE']

X_train=X_train[select_var]
X_test=X_test[select_var]

decision_tree_classification()
# # Hyper Tuning : Randomized Search CV (Random Forest)
#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 200,stop = 1000, num = 5)]

#Number of features to consider while splitting
max_features = ['auto', 'sqrt']

#Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(start = 7, stop = 10, num = 4)]

#Minimum # of samples required to split the node
min_samples_split = [10,15]

#Minimum # of samples required at each leaf node
min_samples_leaf = [3,5]

#Method of selecting samples from training each tree
bootstrap = [True, False]

scoring={'AUC' : make_scorer(roc_auc_score) , 
         'Accuracy':make_scorer(accuracy_score), 
         'Recall':make_scorer(recall_score), 
         'Precision':make_scorer(precision_score),
         'F1 Score':make_scorer(f1_score)}

#Create the random grid:
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf ,
               'bootstrap': bootstrap                
              }

RFcl = RandomForestClassifier(random_state = 0, n_jobs = -1) 

#CV_rfc = RandomizedSearchCV(estimator=RFcl, param_distributions =random_grid, n_jobs = -1, cv= 10,scoring=scoring,refit='Recall',return_train_score=True,n_iter=10)
CV_rfc = RandomizedSearchCV(estimator=RFcl, param_distributions =random_grid, n_jobs = -1, cv= 10,scoring=scoring,refit='AUC',return_train_score=True,n_iter=10)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.cv_results_)


# # Hyper Tuning : Grid Search CV (Random Forest)
def format_grid_search_result(res):
    global df_gs_result
    gs_results=res
    
    gs_model=gs_results['params']
    
    # Grid Search : AUC Metrics
    gs_mean_test_AUC=pd.Series(gs_results['mean_test_AUC'])
    gs_std_test_AUC=pd.Series(gs_results['std_test_AUC'])
    gs_rank_test_AUC=pd.Series(gs_results['rank_test_AUC'])
    
    # Grid Search : Accuracy Metrics
    gs_mean_test_Accuracy=pd.Series(gs_results['mean_test_Accuracy'])
    gs_std_test_Accuracy=pd.Series(gs_results['std_test_Accuracy'])
    gs_rank_test_Accuracy=pd.Series(gs_results['rank_test_Accuracy'])
    
    # Grid Search : Recall Metrics
    gs_mean_test_Recall=pd.Series(gs_results['mean_test_Recall'])
    gs_std_test_Recall=pd.Series(gs_results['std_test_Recall'])
    gs_rank_test_Recall=pd.Series(gs_results['rank_test_Recall'])

    # Grid Search : Precision Metrics
    gs_mean_test_Precision=pd.Series(gs_results['mean_test_Precision'])
    gs_std_test_Precision=pd.Series(gs_results['std_test_Precision'])
    gs_rank_test_Precision=pd.Series(gs_results['rank_test_Precision'])
    
    # Grid Search : F1-Score Metrics
    gs_mean_test_F1_Score=pd.Series(gs_results['mean_test_F1 Score'])
    gs_std_test_F1_Score=pd.Series(gs_results['std_test_F1 Score'])
    gs_rank_test_F1_Score=pd.Series(gs_results['rank_test_F1 Score'])   

    
    gs_model_split=str(gs_model).replace("[{","").replace("}]","").split('}, {')
    df_gs_result=pd.DataFrame(gs_model_split,index=None,columns=['Model_attributes'])
    df_gs_result=pd.concat([df_gs_result,gs_mean_test_AUC,gs_std_test_AUC,gs_rank_test_AUC,gs_mean_test_Accuracy,gs_std_test_Accuracy,gs_rank_test_Accuracy,gs_mean_test_Recall,gs_std_test_Recall,gs_rank_test_Recall,gs_mean_test_Precision,gs_std_test_Precision,gs_rank_test_Precision,gs_mean_test_F1_Score,gs_std_test_F1_Score,gs_rank_test_F1_Score],axis=1)
    
    df_gs_result.columns=['Model_attributes','mean_test_AUC','std_test_AUC','rank_test_AUC','mean_test_Accuracy','std_test_Accuracy','rank_test_Accuracy','mean_test_Recall','std_test_Recall','rank_test_Recall','mean_test_Precision','std_test_Precision','rank_test_Precision','mean_test_F1_Score','std_test_F1_Score','rank_test_F1_Score']  
    
#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 200,stop = 1000, num = 5)]

#Number of features to consider while splitting
max_features = ['auto', 'sqrt']

#Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(start = 7, stop = 10, num = 4)]

#Minimum # of samples required to split the node
min_samples_split = [10,15]

#Minimum # of samples required at each leaf node
min_samples_leaf = [3,5]

#Method of selecting samples from training each tree
bootstrap = [True, False]

scoring={'AUC' : make_scorer(roc_auc_score) , 
         'Accuracy':make_scorer(accuracy_score), 
         'Recall':make_scorer(recall_score), 
         'Precision':make_scorer(precision_score),
         'F1 Score':make_scorer(f1_score)}

#Create the grid:
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf ,
               'bootstrap': bootstrap                
              }

  
RFcl = RandomForestClassifier(random_state = 0, n_jobs = -1) 

GS_rfc = GridSearchCV(estimator=RFcl, param_grid=random_grid, cv= 10, n_jobs = -1,scoring=scoring,refit='Recall',return_train_score=True)
GS_rfc.fit(X_train, y_train)
print(GS_rfc.best_score_)
    
format_grid_search_result(GS_rfc.cv_results_)
df_gs_result.to_excel('Random_forest_Grid_Search_24feb.xlsx')




# # Random Forest : Optimized Model
def random_forest_classification():
    global X_train,y_train,X_test,y_test
    global clf
    
    from sklearn.ensemble import RandomForestClassifier

    # Grid Search - Least Standard Deviation :'bootstrap': True, 'max_depth': 9, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 200
   

    clf = RandomForestClassifier(bootstrap= True, max_depth= 9, max_features= 'auto', min_samples_leaf= 5, min_samples_split=10, n_estimators=200, verbose = 1, n_jobs = -1,random_state=100)

    
   
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_train)
    
    print("====== Random_Forest_Classification  Train ======")
    print(" Accuracy : "  + str(metrics.accuracy_score(y_train,y_pred)))
    print(" Recall : "  + str(metrics.recall_score(y_train,y_pred)))
    print(" Precision : "  + str(metrics.precision_score(y_train,y_pred)))
    print(" F1_Score : "  + str(metrics.f1_score(y_train,y_pred)))
    print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_train,y_pred)))
    print(" ")

    y_pred = clf.predict(X_test)
    
    print("====== Random_Forest_Classification Test ======")
    print(" Accuracy : "  + str(metrics.accuracy_score(y_test,y_pred)))
    print(" Recall : "  + str(metrics.recall_score(y_test,y_pred)))
    print(" Precision : "  + str(metrics.precision_score(y_test,y_pred)))
    print(" F1_Score : "  + str(metrics.f1_score(y_test,y_pred)))
    print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_test,y_pred)))
    print(" ")

       
select_var=['Pclass','Child_ind','Fare','Sex_WOE','Dependent_ind_WOE','Embarked_ind_WOE']

X_train=X_train[select_var]
X_test=X_test[select_var]

random_forest_classification()

# # Random forest Importance : Final Variables
feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances
# # Hyper Tuning : Grid Search CV (Gradient Boosting Machine)

#Number of tress in Random Forest
n_estimators = [int(x) for x in np.linspace(start = 200,stop = 1000, num = 5)]

#Number of features to consider while splitting
max_features = ['auto', 'sqrt']

#Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(start = 7, stop = 10, num = 4)]

#Minimum # of samples required to split the node
min_samples_split = [10,15]

#Minimum # of samples required at each leaf node
min_samples_leaf = [3,5]


from sklearn.metrics import accuracy_score,make_scorer,precision_score,recall_score,roc_auc_score,f1_score
scoring={'AUC' : make_scorer(roc_auc_score) , 
         'Accuracy':make_scorer(accuracy_score), 
         'Recall':make_scorer(recall_score), 
         'Precision':make_scorer(precision_score),
         'F1 Score':make_scorer(f1_score)}

#Create the random grid:
gbm_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf 
              }

gbm_cl = GradientBoostingClassifier(random_state = 0) 

GS_gbm = GridSearchCV(estimator=gbm_cl, param_grid=gbm_grid, cv= 10, n_jobs = -1,scoring=scoring,refit='Recall',return_train_score=True)
GS_gbm.fit(X_train, y_train)
format_grid_search_result(GS_gbm.cv_results_)
df_gs_result.to_excel('Gradient_Boosting_Machine_Grid_Search_24feb.xlsx')


# # Gradient Boosting Machine : Optimized Model
def gradient_boost_classification():
    global X_train,y_train,X_test,y_test
    global gbm
  
    from sklearn.ensemble import GradientBoostingClassifier
    
    learn_list=[0.05]
    
    # Grid Search - Least Standard Deviation for Recall : 'max_depth': 9, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 200
  
    
    for lrn_rate in learn_list:
        gbm = GradientBoostingClassifier(n_estimators=200,learning_rate=lrn_rate,max_features='sqrt',max_depth=9,min_samples_leaf=3,min_samples_split=10,subsample=0.8,random_state=0)

        
        gbm.fit(X_train,y_train)
       
        y_pred = gbm.predict(X_train[select_var])
        gradient_boost_model_accuracy = metrics.accuracy_score(y_train,y_pred)
        print("====== Gradient_Boosting_Classification for learning Rate (train) : " + str(lrn_rate))
        print(" Accuracy : "  + str(metrics.accuracy_score(y_train,y_pred)))
        print(" Recall : "  + str(metrics.recall_score(y_train,y_pred)))
        print(" Precision : "  + str(metrics.precision_score(y_train,y_pred)))
        print(" F1_Score : "  + str(metrics.f1_score(y_train,y_pred)))
        print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_train,y_pred)))
        print(" ")

        y_pred = gbm.predict(X_test)
    
        print("====== Gradient_Boosting_Classification for learning Rate(test) : " + str(lrn_rate))
        print(" Accuracy : "  + str(metrics.accuracy_score(y_test,y_pred)))
        print(" Recall : "  + str(metrics.recall_score(y_test,y_pred)))
        print(" Precision : "  + str(metrics.precision_score(y_test,y_pred)))
        print(" F1_Score : "  + str(metrics.f1_score(y_test,y_pred)))
        print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_test,y_pred)))
        print(" ")

          
select_var=['Pclass','Child_ind','Fare','Sex_WOE','Dependent_ind_WOE','Embarked_ind_WOE']

X_train=X_train[select_var]
X_test=X_test[select_var]

gradient_boost_classification()
#Lift Gain Chart
select_var=['Pclass','Child_ind','Fare','Sex_WOE','Dependent_ind_WOE','Embarked_ind_WOE']

X_train=X_train[select_var]
X_test=X_test[select_var]


y_pred = gbm.predict(X_train)
y_train_score = gbm.predict_proba(X_train)
random_forest_model_accuracy = metrics.accuracy_score(y_train,y_pred)
                
print("====== Classification Metrics - Development ======")
print(" Accuracy : "  + str(metrics.accuracy_score(y_train,y_pred)))
print(" Recall : "  + str(metrics.recall_score(y_train,y_pred)))
print(" Precision : "  + str(metrics.precision_score(y_train,y_pred)))
print(" F1_Score : "  + str(metrics.f1_score(y_train,y_pred)))
print(" Confusion_metrics : "  + str(metrics.confusion_matrix(y_train,y_pred)))
print(" ")

y_train_score_df = pd.DataFrame(y_train_score, index=range(y_train_score.shape[0]),columns=range(y_train_score.shape[1]))
y_train_score_df['Actual'] = pd.Series(y_train.values, index=y_train_score_df.index)
y_train_score_df['Predicted'] = pd.Series(y_pred, index=y_train_score_df.index)
y_train_score_df['Decile'] = pd.qcut(y_train_score_df[1],10,duplicates='drop')

lift_tbl = pd.DataFrame([y_train_score_df.groupby('Decile')[1].min(),
                                                 y_train_score_df.groupby('Decile')[1].max(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 1)].groupby('Decile')[1].count(),
                                                 y_train_score_df[(y_train_score_df['Actual'] == 0)].groupby('Decile')[1].count(),
                                                 y_train_score_df.groupby('Decile')[1].count()]).T
lift_tbl.columns = ["MIN","MAX","Event","Non-Event","TOTAL"]
lift_tbl = lift_tbl.sort_values("MIN", ascending=False)
lift_tbl = lift_tbl.reset_index()

list_vol_pct=[]
list_event_pct=[]

for i in range(len(lift_tbl.Event)):
    list_vol_pct.append(lift_tbl['TOTAL'][i]/lift_tbl['TOTAL'].sum())
    list_event_pct.append(lift_tbl['Event'][i]/lift_tbl['TOTAL'][i])

lift_tbl = pd.concat([lift_tbl,pd.Series(list_vol_pct),pd.Series(list_event_pct)],axis=1)


lift_tbl = lift_tbl[["Decile","MIN","MAX","Event","Non-Event","TOTAL",0,1]]        
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-2]: "Volume(%)"})
lift_tbl = lift_tbl.rename(columns={lift_tbl.columns[len(lift_tbl.keys())-1]: "Event(%)"})

lift_tbl["Cumm_Event"] = lift_tbl["Event"].cumsum()
lift_tbl["Cumm_Event_Pct"] = lift_tbl["Cumm_Event"] / lift_tbl["Event"].sum()
#lift_tbl
lift_tbl.to_excel("Titanic_Lift_Chart_optimized_gbm_24feb2020.xlsx", index = None, header=True)
lift_tbl


# # ROC Graph for the Model
# SHOW ROC / AUC PLOT 
class CurvePlotter():
    def __init__(self):
        self.y_test_list = []
        self.prob_series_list = []
        self.tprs = [] # list to hold true / false positives
        self.aucs = [] # to hold AUC values
        
        self.count = 1 # counter
        self.mean_fpr = np.linspace(0, 1, 100)
         
    def add_to_y_test_list(self, y_test):
        self.y_test_list.append(y_test)
        
    def add_to_prob_series_list(self, prob_series):
        self.prob_series_list.append(prob_series)
        # add logic ot check if two or 1 and handle
        
    def loop_through_plot_values(self):
        for y_test, prob_series in zip(self.y_test_list, self.prob_series_list):
            # calculate roc curve
            fpr, tpr, thresholds = roc_curve(y_test, prob_series)
            # append interpolated values
            self.tprs.append(interp(self.mean_fpr, fpr, tpr))
            # set beginning value of last list entry to 0
            self.tprs[-1][0] = 0.0
            # calculate auc
            roc_auc = auc(fpr, tpr)
            # add auc to list
            self.aucs.append(roc_auc)
            # add to plot object
            plt.plot(fpr, tpr, lw = 1, alpha = 0.3, label = 'ROC fold %d (AUC = %0.2f)' % (self.count,roc_auc))
            
            self.count += 1
    
    def generate_plots(self):
        #mean_fpr = self.mean_fpr
        # add the reference "chance" line
        plt.plot([0,1], [0,1], linestyle='--',lw=2, color = 'r', label='Chance',alpha=.8)
        # calculate mean of list
        mean_tpr = np.mean(self.tprs,axis = 0)
        # set last point = 1
        mean_tpr[-1] = 1 
        # calculate mean AUC
        mean_auc = auc(self.mean_fpr, mean_tpr)
        # calculate standard deviation of aucs
        std_auc = np.std(self.aucs)
        
        plt.plot(self.mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
        
        std_tpr = np.std(self.tprs, axis=0)
        
        # calculate upper and lower bounds of true positives
        tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
            
        # set display settings for plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic graph')
        plt.legend(loc="lower right")
        
    def get_plot(self):
        self.loop_through_plot_values()
        self.generate_plots()
        return plt


curve = CurvePlotter()    

# GET THE PROBABILITIES ([PROB_NO, PROB_YES])
probs = gbm.predict_proba(X_train) # series of predicted probabilties

# add data for creating ROC / AUC graph
curve.add_to_y_test_list(y_train)

curve.add_to_prob_series_list(probs[:,1])
plt = curve.get_plot()

from sklearn.metrics import roc_auc_score
y_train_score = gbm.predict_proba(X_train)
print("ROC AUC Score (GBM Model Development) : " + str(roc_auc_score(y_train,y_train_score[:,1])))
plt.show()





