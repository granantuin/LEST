import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
from io import BytesIO
import base64
import pickle
import warnings
import streamlit as st
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


warnings.filterwarnings("ignore")

st.set_page_config(page_title="Santiago de Compostela airport Machine Learning forecast",layout="wide")

def Hss(cm):
     """
     obtain de Heidke skill score from a 3x3 confusion matrix (margins=on)
     
     Returns: Heidke skill score
     """
     if cm.shape == (3,3):
          a = cm.values[0,0]
          b = cm.values[1,0]
          c = cm.values[0,1]
          d = cm.values[1,1]
          HSS = round(2*(a*d-b*c)/((a+c)*(c+d)+(a+b)*(b+d)),2)
     else:
          HSS = 0
     return HSS


def get_metar(oaci,control):
     """
     get metar from IOWA university database
     
     in: OACI airport code
     Returns
      -------
     dataframe with raw metar.
     """
     #today metar control =True
     if control:
       today = pd.to_datetime("today")+timedelta(1)
       yes = today-timedelta(1)
     else:
        today = pd.to_datetime("today")+timedelta(1)
        yes = today-timedelta(2)

     #url string
     s1="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station="
     s2="&data=all"
     s3="&year1="+yes.strftime("%Y")+"&month1="+yes.strftime("%m")+"&day1="+yes.strftime("%d")
     s4="&year2="+today.strftime("%Y")+"&month2="+today.strftime("%m")+"&day2="+today.strftime("%d")
     s5="&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"
     url=s1+oaci+s2+s3+s4+s5
     df_metar_global=pd.read_csv(url,parse_dates=["valid"],).rename({"valid":"time"},axis=1)
     df_metar = df_metar_global[["time",'tmpf', 'dwpf','drct', 'sknt', 'alti','vsby',
                                 'gust', 'skyc1', 'skyc2', 'skyl1', 'skyl2','wxcodes',
                                 "metar"]].set_index("time")
     
     #temperature dry a dew point to celsius                            
     df_metar["temp_o"] = np.rint((df_metar.tmpf - 32)*5/9)
     df_metar["tempd_o"] = np.rint((df_metar.dwpf - 32)*5/9)

     #QNH to mb
     df_metar["mslp_o"] = np.rint(df_metar.alti*33.8638)

     #visibility SM to meters
     df_metar["visibility_o"] =np.rint(df_metar.vsby/0.00062137)

     #wind direction, intensity and gust
     df_metar["spd_o"] = df_metar["sknt"]
     df_metar["dir_o"] = df_metar["drct"]
     df_metar['gust_o'] = df_metar['gust'] 

     #Add suffix cloud cover and cloud height, present weather, and metar 
     df_metar['skyc1_o'] = df_metar['skyc1']
     df_metar["skyl1_o"] = df_metar["skyl1"]
     df_metar['skyc2_o'] = df_metar['skyc2']
     df_metar["skyl2_o"] = df_metar["skyl2"]
     df_metar["wxcodes_o"] = df_metar["wxcodes"]
     df_metar["metar_o"] = df_metar["metar"]
     
     # Select all columns that do not start with "_o"
     columns_to_keep = [col for col in df_metar.columns if col.endswith("_o")]
     df_metar = df_metar[columns_to_keep] 

     return df_metar 
  
def get_meteogalicia_model_4Km(coorde):
    """
    get meteogalicia model (4Km)from algo coordenates
    Returns
    -------
    dataframe with meteeorological variables forecasted.
    """
    
    #defining url to get model from Meteogalicia server
    var1 = "var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
    var2 = "&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=T850"
    var3 = "&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=snow_prec&var=snowlevel"
    var = var1+var2+var3 
    head1="https://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d03"
    
    
    #url12="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d02/2016/09/wrf_arw_det_history_d02_20160927_0000.nc4?var=mod&disableLLSubset=on&dis
    try:
          
      today = pd.to_datetime("today")    
      head2 = today.strftime("/%Y/%m/wrf_arw_det_history_d03")
      head3 = today.strftime("_%Y%m%d_0000.nc4?")
      head = head1+head2+head3
       
      f_day=(today+timedelta(days=2)).strftime("%Y-%m-%d") 
      tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"
  
      dffinal=pd.DataFrame() 
      for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
          dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)    
  
      #filter all columns with lat lon and date
      dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')
  
      #remove column string between brakets
      new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
      for col in zip(dffinal.columns,new_col):
          dffinal=dffinal.rename(columns = {col[0]:col[1]})
  
      dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=3)).strftime("%Y-%m-%d"), freq="H")[1:-1])  
      control = True
          
    except:

      today = pd.to_datetime("today")-timedelta(1)
      head2 = today.strftime("/%Y/%m/wrf_arw_det_history_d03")
      head3 = today.strftime("_%Y%m%d_0000.nc4?")
      head = head1+head2+head3
        
      f_day=(today+timedelta(days=2)).strftime("%Y-%m-%d") 
      tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"
  
      dffinal=pd.DataFrame() 
      for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
          dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)    
  
      
      #filter all columns with lat lon and date
      dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')
  
      #remove column string between brakets
      new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
      for col in zip(dffinal.columns,new_col):
          dffinal=dffinal.rename(columns = {col[0]:col[1]})
  
      dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=3)).strftime("%Y-%m-%d"), freq="H")[1:-1])  
      control= False  

     
    return dffinal , control


#score machine learning versus WRF
score_ml = 0
score_wrf = 0
best_ml = []
best_wrf = []

# Set the directory you want to list algorithms filenames from
algo_dir = 'algorithms/'

#get meteorological model from algorithm file. Select "coor" key to get coordinates. Pick up first algorithm all same coordinates
#meteo_model,con = get_meteogalicia_model_4Km(pickle.load(open(algo_dir+os.listdir(algo_dir)[0],"rb"))["coor"])
meteo_model,con = get_meteogalicia_model_4Km(pickle.load(open("algorithms/dir_LEST_d0.al","rb"))["coor"])
#add time variables
meteo_model["hour"] = meteo_model.index.hour
meteo_model["month"] = meteo_model.index.month
meteo_model["dayofyear"] = meteo_model.index.dayofyear
meteo_model["weekofyear"] = meteo_model.index.isocalendar().week.astype(int)

#show meteorological model and control variable. Control variable True if Day analysis = today 
#st.write(#### **Day analysis = today :**",con)
#st.write(meteo_model)

metars = get_metar("LEST",con)
st.markdown(" ### **Metars**")
AgGrid(metars[["metar_o","dir_o","spd_o","gust_o","visibility_o","wxcodes_o","skyc1_o","skyl1_o","skyc2_o","skyl2_o","temp_o","tempd_o","mslp_o"]])


#@title Wind direction

#open algorithm spd d0 d1
alg = pickle.load(open("algorithms/dir_LEST_d0.al","rb"))
alg1 = pickle.load(open("algorithms/dir_LEST_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat spd from ml
dir_ml = alg["pipe"].predict(model_x_var)
dir_ml1 = alg1["pipe"].predict(model_x_var1)

#set up dataframe forecast machine learning and WRF
df_for = pd.DataFrame({"time":meteo_model[:48].index,
                        "dir_WRF": np.concatenate((model_x_var["dir0"],model_x_var1["dir0"]),axis=0),
                        "dir_ml": np.concatenate((dir_ml,dir_ml1),axis =0),})
df_for = df_for.set_index("time")

#label dir_o and dir0 .wind direction to interval dir=-1 variable wind
interval = pd.IntervalIndex.from_tuples([(-1.5, -0.5),(-0.5,20), (20, 40), (40, 60),
                                           (60,80),(80,100),(100,120),(120,140),(140,160),
                                           (160,180),(180,200),(200,220),(220,240),
                                           (240,260),(260,280),(280,300),(300,320),
                                           (320,340),(340,360)])
labels = ['VRB', '[0, 20]', '(20, 40]', '(40, 60]','(60, 80]', '(80, 100]',
          '(100, 120]', '(120, 140]','(140, 160]', '(160, 180]', '(180, 200]',
          '(200, 220]','(220, 240]', '(240, 260]', '(260, 280]', '(280, 300]',
          '(300, 320]', '(320, 340]', '(340, 360]']
df_for["dir_WRF_l"] = pd.cut(df_for["dir_WRF"], bins=interval,retbins=False,
                        labels=labels).map({a:b for a,b in zip(interval,labels)}).astype(str)

#dir_o to intervals 
metars["dir_o_l"] = pd.cut(metars["dir_o"].replace("M",-1).astype(float), bins=interval,retbins=False,
                        labels=labels).map({a:b for a,b in zip(interval,labels)}).astype(str)                    

# concat metars an forecast
df_res = pd.concat([df_for,metars[["dir_o","dir_o_l"]]],axis = 1)

#get accuracy
df_res_dropna = df_res.dropna()
acc_ml = round(accuracy_score(df_res_dropna.dir_o_l,df_res_dropna.dir_ml),2)
acc_wrf = round(accuracy_score(df_res_dropna.dir_o_l,df_res_dropna.dir_WRF_l),2)
if acc_ml>acc_wrf:
  score_ml+=1
  best_ml.append("wind direction")   
if acc_ml<acc_wrf:  
  score_wrf+=1
  best_wrf.append("wind direction")   

#Show results
st.markdown(" #### **Wind direction**")
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_res_dropna.index, df_res_dropna['dir_ml'], marker="^", markersize=8, 
         markerfacecolor='w', color="b", linestyle='')
plt.plot(df_res_dropna.index, df_res_dropna['dir_o_l'], marker="*", markersize=13,
         markerfacecolor='k', color= "g", linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['dir_WRF_l'], marker="v", markersize=8,
         markerfacecolor='w', color="r", linestyle='');
plt.legend(('direction ml', 'direction observed', 'direction WRF'),)
plt.grid(True, axis="both", which="both")
plt.title("Actual accuracy meteorological model: {:.0%}. Reference: 25%\nActual accuracy machine learning: {:.0%}. Reference: 40%".format(acc_wrf,acc_ml))
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(df_for.index, df_for['dir_ml'],marker="^", color="b", linestyle='');
plt.plot(df_for.index, df_for['dir_WRF_l'],marker="v",color="r", linestyle='');
plt.legend(('direction ml','direction WRF'),)
plt.title("Forecast meteorological model versus machine learning")
plt.grid(True)
st.pyplot(fig)

#probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = pd.DataFrame(prob,index =alg["pipe"].classes_ ).T
df_prob = df_prob[labels]
df_prob.index = meteo_model[:48].index.strftime('%b %d %H:%M Z')

# Find the columns where all values are less than or equal to 5%
cols_to_drop = df_prob.columns[df_prob.apply(lambda x: x <= 0.05).all()]
df_prob.drop(cols_to_drop, axis=1, inplace=True)

#Display
fig1, ax = plt.subplots()
sns.heatmap(df_prob[:48], annot=True, cmap='coolwarm',
            linewidths=.2, linecolor='black',fmt='.0%',
           annot_kws={'size': 5})
plt.title('Probabilities wind direction more than 5%')
st.pyplot(fig1)


#@title Wind intensity

#open algorithm spd d0 d1
alg = pickle.load(open("algorithms/spd_LEST_d0.al","rb"))
alg1 = pickle.load(open("algorithms/spd_LEST_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat spd from ml and wrf
spd_ml = alg["pipe"].predict(meteo_model[:24][alg["x_var"]])
spd_ml1 = alg1["pipe"].predict(meteo_model[24:48][alg1["x_var"]])
df_for = pd.DataFrame({"time":meteo_model[:48].index,
                       "spd_WRF": np.concatenate((np.rint(model_x_var["mod0"]*1.94384),
                                                   np.rint(model_x_var1["mod0"]*1.94384)),axis=0),
                       "spd_ml": np.concatenate((np.rint(spd_ml*1.94384),
                                                  np.rint(spd_ml1*1.94384)),axis =0),})
df_for = df_for.set_index("time")

# concat metars an forecast
df_res = pd.concat([df_for,metars["spd_o"]],axis = 1)

#get mae
df_res_dropna = df_res.dropna()
mae_ml = round(mean_absolute_error(df_res_dropna.spd_o,df_res_dropna.spd_ml),2)
mae_wrf = round(mean_absolute_error(df_res_dropna.spd_o,df_res_dropna.spd_WRF),2)
if mae_ml < mae_wrf:
  score_ml+=1
  best_ml.append("wind speed")   
if mae_ml > mae_wrf:  
  score_wrf+=1
  best_wrf.append("wind speed")
     
#show results actual versus models
st.markdown(" ### **Wind intensity knots**")
fig, ax = plt.subplots(figsize=(8,6))
df_res.dropna().plot(grid = True, ax=ax, linestyle='--', color = ["r","b","g"]);
title = "Actual mean absolute error meteorological model (kt): {}. Reference (m/s): 1.1\nActual mean absolute error machine learning (kt): {}. Reference (m/s): 0.68".format(mae_wrf,mae_ml)
ax.set_title(title)
ax.grid(True, which = "both", axis = "both")
st.pyplot(fig)

# show forecasts
fig, ax = plt.subplots(figsize=(8,6))
df_for.plot(grid=True, ax=ax, color= ["r","b"],linestyle='--')
ax.set_title("Forecast meteorological model versus machine learning")
ax.grid(True, which = "both", axis = "both")
st.pyplot(fig)


#@title BR or FG
#open algorithm prec d0 d1
alg = pickle.load(open("algorithms/brfg_LEST_d0.al","rb"))
alg1 = pickle.load(open("algorithms/brfg_LEST_d1.al","rb"))

#select model variables
model_x_var = meteo_model[:24][alg["x_var"]]
model_x_var1 = meteo_model[24:48][alg1["x_var"]]

# forecat br/fg from ml
brfg_ml = alg["pipe"].predict(model_x_var)
brfg_ml1 = alg1["pipe"].predict(model_x_var1)

#label metars br/fg data
metars["brfg_o_l"] = "No BR/FG"
mask = metars['wxcodes_o'].str.contains("BR")
metars.loc[mask,["brfg_o_l"]] = "BR/FG"
mask = metars['wxcodes_o'].str.contains("FG")
metars.loc[mask,["brfg_o_l"]] = "BR/FG"

#set up dataframe forecast machine learning 
df_for = pd.DataFrame({"time": meteo_model[:48].index,
                       "brfg_ml": np.concatenate((brfg_ml,brfg_ml1),axis =0),})
df_for = df_for.set_index("time")

# concat metars an forecast
df_res = pd.concat([df_for,metars["brfg_o_l"]], axis = 1)
df_res_dropna = df_res.dropna()

#Heidke skill score ml
cm_ml = pd.crosstab(df_res.dropna().brfg_o_l, df_res.dropna().brfg_ml, margins=True,)
acc_ml = round(accuracy_score(df_res_dropna.brfg_o_l,df_res_dropna.brfg_ml),2)
HSS_ml = Hss(cm_ml)

#show results
st.markdown(" ### **BR or FG**")
fig1, ax = plt.subplots(figsize=(4,2))
sns.heatmap(cm_ml, annot=True, cmap='coolwarm',
            linewidths=.2, linecolor='black',)
plt.title("Confusion matrix\nAccuracy machine learning: {:.0%}".format(acc_ml))
st.pyplot(fig1)

fig, ax = plt.subplots(figsize=(10,4))
plt.plot(df_res_dropna.index, df_res_dropna['brfg_ml'],marker="^", markersize=8, 
         markerfacecolor='w', color="b",linestyle='');
plt.plot(df_res_dropna.index, df_res_dropna['brfg_o_l'],marker="*",markersize=8, 
         markerfacecolor='w', color="g",linestyle='');
plt.legend(('brfg ml', 'brfg observed'),)
plt.grid(True,axis="both")
plt.title("Actual Heidke skill score machine learning: {}. Reference: 0.64".format(HSS_ml))
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,4))
plt.plot(df_for.index, df_for['brfg_ml'],marker="^",linestyle='');
plt.title("Forecast machine learning")
plt.grid(True,axis="both")
st.pyplot(fig)

#show probabilistic results
prob = (np.concatenate((alg["pipe"].predict_proba(model_x_var),alg1["pipe"].predict_proba(model_x_var1)),axis =0)).transpose()
df_prob = (pd.DataFrame(prob,index =alg["pipe"].classes_ ).T.set_index(meteo_model[:48].index.map(lambda t: t.strftime('%d-%m %H'))))
fig, ax = plt.subplots(figsize=(10,8))
df_prob["BR/FG"] = df_prob["BR/FG"].round(1)
df_prob["BR/FG"].plot(ax = ax, grid = True, ylim =[0, 1], title = "BR or FG probability", kind='bar')
st.pyplot(fig)


#global results
st.write("#### **Global results**")
st.write("Better meteorological model outcome: {}".format(score_wrf))
st.write(best_wrf)
st.write("Better machine learning outcome: {}".format(score_ml))
st.write(best_ml)


st.write("Project [link](https://github.com/granantuin/LECO)")

