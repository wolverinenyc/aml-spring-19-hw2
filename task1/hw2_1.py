import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pylab import savefig

#   PARCEL IDENTIFICATION NUMBER CAN BE USED WITH CITY WEBSITE, SO USE THAT AS INDEX, DROP ORDER

df = pd.read_excel('AmesHousing.xls',sheet_name='Sheet1',index_col=1)
df = df.drop(['Order'],axis=1)

contvarlist = ['Lot Frontage','Lot Area','Mas Vnr Area','BsmtFin SF 1','BsmtFin SF 2','Bsmt Unf SF',
               'Total Bsmt SF', '1st Flr SF','2nd Flr SF','Low Qual Fin SF','Gr Liv Area','Garage Area',
               'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch','3Ssn Porch','Screen Porch','Pool Area',
               'Misc Val']

nominalvarlist = ['MS SubClass','MS Zoning','Street','Alley','Land Contour','Lot Config','Neighborhood','Condition 1',
                  'Condition 2','Bldg Type','House Style','Roof Style','Roof Matl','Exterior 1st',
                  'Exterior 2nd','Mas Vnr Type','Foundation','Heating','Central Air','Garage Type',
                  'Misc Feature','Sale Type','Sale Condition']

ordinalvarlist = ['Lot Shape','Utilities','Land Slope','Exter Qual','Exter Cond',
                  'Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1', 'BsmtFin Type 2','Heating QC',
                  'Electrical','Kitchen Qual','Functional','Fireplace Qu','Garage Finish','Garage Qual',
                  'Garage Cond','Paved Drive','Pool QC','Fence']

discretevarlist = ['Year Built','Year Remod/Add','Bsmt Full Bath','Bsmt Half Bath','Full Bath','Half Bath','Bedroom AbvGr',
                   'Kitchen AbvGr','TotRms AbvGrd','Fireplaces','Garage Yr Blt','Garage Cars','Mo Sold',
                   'Yr Sold','Overall Qual','Overall Cond']

yearvarlist = ['Year Built','Year Remod/Add','Garage Yr Blt','Mo Sold','Yr Sold']


#   TASK 1.1
#   VISUALIZE DISTRIBUTION OF EACH CONTINUOUS VARIABLE AND DISTRIBUTION OF TARGET
#   NOTE THAT SOME NUMERICAL VARIABLES ARE ACTUALLY CATEGORICAL

task11list = contvarlist + ['SalePrice']
contdf = df[task11list]

#   BESIDES TARGET, THERE ARE 36 NUMERICAL VARIABLES. CAN VISUALIZE IN 6x6 SUBPLOT

fig, ax = plt.subplots(5,4,figsize=(25,15))

for i in range(4):
    for j in range(5):
        colnum = 5*i + j
        ax[j,i].hist(contdf.iloc[:,colnum])
        ax[j,i].set_xlabel(contdf.columns[colnum])
            
plt.figtext(0.4,0.92,'Task 1.1: Visualize Distribution of Continuous Variables',
            fontsize=16)

savefig('hw2_task1_1_continuous_variables.png',bbox_inches='tight')


#   DATA NOTES:  SOME NUMERIC VARIABLES ARE CATEGORICAL (E.G. MS SUBCLASS, # BATHROOMS, FIREPLACES)
#   OTHERS BUNCH AT 0: E.G. 2ND FLOOR SQFT OR PORCH SIZE (FOR THOSE WITHOUT 2ND FLOOR OR PORCH).
#   VARIABLES SUCH AS YEAR/MONTH SOLD CAN BE GROUPED, ALSO CONSIDER SEASONAL EFFECTS (E.G. HIGHER SALES PRICES IN SUMMER)
#   NEED TO CONSIDER THAT SOME VARIABLES HIGHLY RELATED: E.G. IF NO GARAGE, THEN ALL THE NOMINAL AND ORDINAL
#   GARAGE VARIABLES HAVE VALUE 'NA' WHILE THE CONTINOUS GARAGE VARIABLES HAVE VALUE '0'

#   DISCRETE VARIABLES COULD BE TREATED AS CATEGORIES OR MODELED AS IF CONTINUOUS


#   TASK 1.2
#   VISUALIZE 2-D SCATTER DEPENDENCIES OF TARGET ON CONTINUOUS VARS

y = df['SalePrice']

fig, ax = plt.subplots(5,4,figsize=(25,15),sharey='row')

for i in range(4):
    for j in range(5):
        colnum = 5*i + j
        if colnum < 19:
            ax[j,i].scatter(contdf.iloc[:,colnum],y)
            ax[j,i].set_xlabel(contdf.columns[colnum])
        if i==0:
            ax[j,i].set_ylabel('Sale Price')            

plt.figtext(0.4,0.92,'Task 1.2: Visualize Dependency of Sale Price on Continous Vars',
            fontsize=16)

savefig('hw2_task1_2__scatter_continuous_variables.png',bbox_inches='tight')


#   BEFORE SPLITTING, DEAL WITH NAN VALUES SEPARATELY FOR DIFFERENT DATA TYPES
            

#   treat NA in nominal and ordinal as MM, categorical as -9, and continous as 0
#   note: for items like garage yr built, would probably want to set value to zero AND
#   create a dummy for no garage.  For year variables, fill with median value

df1 = df.copy()
df1.loc[:,ordinalvarlist + nominalvarlist] = df1.loc[:, ordinalvarlist + nominalvarlist].fillna('MM')
df1.loc[:,yearvarlist] = df1.loc[:,yearvarlist].fillna(df1.loc[:,yearvarlist].mean())
df1.loc[:,contvarlist + discretevarlist] = df1.loc[:,contvarlist + discretevarlist].fillna(0)


#   TASK 1.3: SPLIT INTO TRAIN-TEST SPLIT. DO NOT USE TEST FOR ANYTHING UNTIL FINAL EVALUATION IN 1.6
#   FOR EACH CATEGORICAL VAR, CROSS VALIDATE A LINEAR REGRESSION MODEL USING JUST THIS VARIABLE (ONE HOT ENCODED).
#   VISUALIZE RELATIONSHIP OF CATEGORICAL VARIABLES THAT PROVIDE BEST R2 WITH TARGET
            
from sklearn.model_selection import train_test_split
X = df1.drop(['SalePrice'],axis=1)
y = df1['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

#   LOOP THROUGH EACH CATEGORICAL VARIABLE. ONE-HOT ENCODE, SO THAT ANY NUMERICAL VARIABLES
#   HAVE TO FIRST BE CONVERTED TO CHARACTERS.  RUN REGRESSIONS, SAVE R^2. VISUALIZE HIGHEST R2

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def get_top_feature(trainX,trainy,featurelist):
    r2list = []
    for v in featurelist:
        tmp=trainX[v]
        tmp2=pd.get_dummies(tmp)
        lr = LinearRegression().fit(tmp2,trainy)
        r2list.append(lr.score(tmp2,trainy))
        
    highr2feature = featurelist[np.argmax(r2list)]
    return highr2feature

def plot_top_feature(trainX,trainy,featurelist):
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    highr2feature = get_top_feature(trainX,trainy,featurelist)
    ax.scatter(trainX[highr2feature],trainy)
    ax.set_xlabel(highr2feature)
    ax.set_ylabel('Sale Price')
    
plot_top_feature(X_train[ordinalvarlist],y_train,ordinalvarlist)
plot_top_feature(X_train[nominalvarlist],y_train,nominalvarlist)
plot_top_feature(X_train[discretevarlist],y_train,discretevarlist)

fig,ax = plt.subplots(1,1,figsize=(15,10))
highr2feature = get_top_feature(X_train,y_train,ordinalvarlist)

ax.scatter(X_train[highr2feature],y_train)
ax.set_xlabel(highr2feature)
ax.set_ylabel('Sale Price')

plt.figtext(0.3,0.92,'Task 1.3: Visualize Sale Price on Best Ordinal Variable Values',
            fontsize=16)

savefig('hw2_task1_3_scatter_best_ordinal_variables.png',bbox_inches='tight')
    
    
#1.4 Use ColumnTransformer and pipeline to encode categorical variables. Evaluate Linear
#Regression (OLS), Ridge, Lasso and ElasticNet using cross-validation with the default
#parameters. Does scaling the data (within the pipeline) with StandardScaler help?
    
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#   SET UP PRE-PROCESSING WITH AND WITHOUT SCALING

catlist = ordinalvarlist + nominalvarlist
preprocess_noscaling = ColumnTransformer(
        [('onehotencoder',OneHotEncoder(handle_unknown='ignore'),catlist)])

preprocess_scaling = ColumnTransformer(
        [('standardscaler',StandardScaler(),contvarlist),
         ('minmaxscaler',MinMaxScaler(),discretevarlist),
         ('onehotencoder',OneHotEncoder(handle_unknown='ignore'),catlist)])


#   FUNCTION THAT INPUTS REGRESSION MODEL AND PREPROCESSING, OUTPUTS MEAN CV SCORE

def get_mean_cv(toscale,estimatorname):
    preprocess = eval("preprocess_"+toscale)
    estimator = eval(estimatorname + "()")
    model = make_pipeline(preprocess,estimator)
    scores = cross_val_score(model,X_train,y_train,cv=10)
    df = pd.DataFrame([[estimatorname,toscale,np.mean(scores)]],columns=['Estimator','Scaling','CV Score'])
    return df

#estimatorlist = [LinearRegression(),Ridge(),Lasso(),ElasticNet()]
#preprocesslist = [preprocess_noscaling,preprocess_scaling]
estimatorlist = ["LinearRegression","Ridge","Lasso","ElasticNet"]
preprocesslist = ["noscaling","scaling"] 
results = pd.DataFrame()
    
for e in estimatorlist:
    for p in preprocesslist:
        results=results.append(get_mean_cv(p,e))

print(results)
results.to_csv('hw2_task1_4_cv_results_estimator_scaling.csv',index=False)


# 1.5: TUNE RESULTS WITH GRIDSEARCHCV
from sklearn.model_selection import GridSearchCV

def get_mean_cv_grid(toscale,estimatorname,param_to_tune,tunelist):
    preprocess = eval("preprocess_"+toscale)
    estimator = eval(estimatorname + "()")
    pipe = make_pipeline(preprocess,estimator)
    param = str(estimatorname).lower()+"__"+param_to_tune
    param_grid = {param : tunelist}
    grid = GridSearchCV(pipe,param_grid,cv=10)
    grid.fit(X_train,y_train)
    df = pd.DataFrame([[estimatorname,toscale,grid.best_score_,grid.best_params_,list(grid.cv_results_['mean_test_score']),tunelist]],
                      columns=['Estimator','Scaling','CV Score','Best Alpha','Param CV','Param List'])
    return df
   
preprocesslist = ["scaling"]
estimatorlist = ["Ridge","Lasso","ElasticNet"]
alphalist = [.01,.1,1.0,10,100]

for e in estimatorlist:
    for p in preprocesslist:
        results=results.append(get_mean_cv_grid(p,e,"alpha",alphalist))

results.to_csv('hw2_task1_5_cv_results_with_tuning.csv',index=False)

#   Visualize dependence of validation scores on paramaters for ridge, lasso, elasticnet

gridresults = results[results['Param CV'].notnull()]

l1 = gridresults[gridresults['Estimator']=="Ridge"].iloc[:,3][0]
l2 = gridresults[gridresults['Estimator']=="Lasso"].iloc[:,3][0]
l3 = gridresults[gridresults['Estimator']=="ElasticNet"].iloc[:,3][0]

fig, ax = plt.subplots(1,1,figsize=(10,5))
lns1 = ax.plot(range(5),l1,linewidth=1.0,color='r',marker='D',ms=4,label='Ridge Tuning')
lns2 = ax.plot(range(5),l2,linewidth=1.0,color='b',marker='D',ms=4,label='Lasso Tuning')
lns3 = ax.plot(range(5),l3,linewidth=1.0,color='g',marker='D',ms=4,label='ElasticNet Tuning')

lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]    
ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5,-0.1),
          fancybox=True,shadow=True,ncol=2)

ax.set_xticks(range(5))
ax.set_xticklabels(['.01','.1','1','10','100'])
ax.set_title('CV Score by Alpha Value')

savefig('hw2_task1_5_cv_score_by_alpha_tuning.png',bbox_inches='tight')
  

#   1.6 VISUALIZE COEFFICIENTS OF THE RESULTING MODELS. DO THEY AGREE ON WHICH FEATURES ARE IMPORTANT?

#   retrain using best parameters and save coefficients

coefs = pd.DataFrame()

pipe = make_pipeline(preprocess_scaling,Ridge(alpha=1.0))
pipe.fit(X_train,y_train)
coefs['Ridge']=pipe.named_steps.ridge.coef_

pipe = make_pipeline(preprocess_scaling,Lasso(alpha=100))
pipe.fit(X_train,y_train)
coefs['Lasso']=pipe.named_steps.lasso.coef_

pipe = make_pipeline(preprocess_scaling,ElasticNet(alpha=.01))
pipe.fit(X_train,y_train)
coefs['Elastic Net']=pipe.named_steps.elasticnet.coef_

coeff_corr = pd.DataFrame(coefs.corr())
coeff_corr.to_csv('hw2_task1_6_correlation_coefficients_by_model.csv')