#########################################################
############### Kickstarter Grading Code ################
#########################################################

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Data import
path = "C:/Users/aruni/Desktop/McGill/INSY 662/Individual Project/"   #This path needs to be updated
df_og_training = pd.read_excel(path+"Kickstarter.xlsx")    # Inputting Kickstarter training data (provided to the students)
df_og_grading = pd.read_excel(path+"Kickstarter-Grading.xlsx")   # Inputting Kickstarter-Grading data for testing

# Data cleaning function
def cleaning(df_eval):
    # Re-arranging order to bring state to the 0th position
    df_eval = df_eval[['state', 'project_id', 'name', 'goal', 'pledged', 'disable_communication', 'country', 'currency',
                       'deadline', 'state_changed_at', 'created_at', 'launched_at', 'staff_pick', 'backers_count',
                       'static_usd_rate', 'usd_pledged', 'category', 'spotlight', 'name_len', 'name_len_clean',
                       'blurb_len', 'blurb_len_clean', 'deadline_weekday', 'state_changed_at_weekday',
                       'created_at_weekday', 'launched_at_weekday', 'deadline_month', 'deadline_day', 'deadline_yr',
                       'deadline_hr', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr',
                       'state_changed_at_hr', 'created_at_month', 'created_at_day', 'created_at_yr', 'created_at_hr',
                       'launched_at_month', 'launched_at_day', 'launched_at_yr', 'launched_at_hr', 'create_to_launch_days',
                       'launch_to_deadline_days', 'launch_to_state_change_days']]

    ### DATA CLEANING ###

    df_clean_grading = df_eval.copy()

    # Cleaning state variable
    df_clean_grading = df_clean_grading[(df_clean_grading["state"] == "failed") | (df_clean_grading["state"] == "successful")]  # only retaining rows where state is successful or failed
    df_clean_grading["state"] = df_clean_grading["state"].map({'successful': 1, 'failed': 0}).astype(int)  # mapping successful as 1, failed as 0

    # Dropping fields identified from the training dataset
    df_clean_grading.drop(['usd_pledged', 'backers_count', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr',
                   'state_changed_at_hr', 'state_changed_at_weekday', 'name', 'name_len', 'name_len_clean', 'blurb_len', 'pledged',
                   'currency', 'static_usd_rate', 'deadline', 'launched_at', 'created_at', 'state_changed_at',
                   'launch_to_state_change_days', 'spotlight', 'disable_communication'], axis=1, inplace=True)

    # Removing NULL values
    df_clean_grading = df_clean_grading.dropna()


    #### HANDLING CATEGORICAL FIELDS ###

    # Creating dummy variables
    dummy_category_grading = pd.get_dummies(df_clean_grading['category'], prefix="category")
    dummy_created_at_weekday_grading = pd.get_dummies(df_clean_grading['created_at_weekday'], prefix="created_at_weekday")
    dummy_deadline_weekday_grading = pd.get_dummies(df_clean_grading['deadline_weekday'], prefix="deadline_weekday")
    dummy_country_grading = pd.get_dummies(df_clean_grading['country'], prefix="country")
    dummy_launched_at_weekday_grading = pd.get_dummies(df_clean_grading['launched_at_weekday'], prefix="launched_at_weekday")

    # Joining dummy fields into the main dataset
    df1_grading = df_clean_grading.copy()
    df1_grading = df1_grading.join(dummy_category_grading)
    df1_grading = df1_grading.join(dummy_created_at_weekday_grading)
    df1_grading = df1_grading.join(dummy_deadline_weekday_grading)
    df1_grading = df1_grading.join(dummy_country_grading)
    df1_grading = df1_grading.join(dummy_launched_at_weekday_grading)

    # Dropping the original categorical fields from the dataset
    df1_grading.drop(['category', 'created_at_weekday', 'deadline_weekday', 'country', 'launched_at_weekday'], axis=1, inplace=True)

    # Dropping 'project_id'
    proj_id_grading = list(df1_grading['project_id'])
    df1_grading.drop(['project_id'], axis=1, inplace=True)

    # Resetting index
    df1_grading.reset_index(drop=True, inplace=True)
    
    return df1_grading

# Cleaning kickstarter and kickstarter-grading datasets by feeding them to the function
df_eval_og = cleaning(df_og_training).copy()
df_eval_df1_grading = cleaning(df_og_grading).copy()

############### Classification Model ###############

a = ['goal', 'staff_pick', 'create_to_launch_days', 'created_at_day', 'created_at_hr', 'deadline_day', 'launched_at_hr',
     'launched_at_day', 'deadline_hr', 'launch_to_deadline_days', 'category_Web', 'blurb_len_clean', 'created_at_month',
     'deadline_month', 'launched_at_month', 'category_Software', 'launched_at_yr', 'created_at_yr', 'deadline_yr',
     'category_Plays', 'launched_at_weekday_Tuesday', 'country_US', 'category_Hardware', 'created_at_weekday_Tuesday',
     'category_Musical', 'created_at_weekday_Monday', 'category_Festivals', 'launched_at_weekday_Monday',
     'created_at_weekday_Wednesday', 'deadline_weekday_Friday', 'created_at_weekday_Thursday', 'launched_at_weekday_Wednesday',
     'created_at_weekday_Friday', 'deadline_weekday_Thursday', 'deadline_weekday_Wednesday', 'launched_at_weekday_Thursday',
     'deadline_weekday_Saturday', 'deadline_weekday_Sunday', 'launched_at_weekday_Friday', 'category_Gadgets',
     'deadline_weekday_Monday', 'created_at_weekday_Sunday', 'country_GB', 'created_at_weekday_Saturday',
     'deadline_weekday_Tuesday', 'category_Apps', 'category_Experimental', 'category_Wearables',
     'launched_at_weekday_Saturday', 'launched_at_weekday_Sunday', 'category_Sound', 'country_CA',
     'category_Robots', 'category_Immersive', 'category_Flight', 'country_AU', 'category_Places', 'category_Spaces',
     'country_DE', 'category_Shorts', 'country_FR', 'category_Makerspaces']

X_grading = df_eval_og[a]
y_grading = df_eval_og["state"]

## Running GBT for selected features from training dataset

X_grading_test = df_eval_df1_grading[a]

from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier()
model = gbt.fit(X_grading, y_grading)
y_grading_test_pred = model.predict(X_grading_test)

#The below two lines of code can be used by the grader to compute accuracy of the final data
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(df_eval_df1_grading['state'], y_grading_test_pred))   



#########################################################
############### Clustering Model Building ###############
#########################################################

# Elbow method to show optimal value of k
df1 = cleaning(df_og_training).copy()
df = df1[['goal', 'staff_pick', 'create_to_launch_days', 'blurb_len_clean', 'launched_at_yr', 'created_at_yr', 'deadline_yr']]

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

labels = km.labels_
silhouette = silhouette_samples(df, labels)
print("Silhoutte Score:",silhouette_score(df, labels))

# Plotting PCA Components 1 and 2
scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df = pca.fit_transform(standardized_x)
reduced_df = pd.DataFrame(df, columns=['PC1','PC2'])
plt.scatter(reduced_df['PC1'], reduced_df['PC2'], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Plotting clusters

kmeans=KMeans(n_clusters=4)
model=kmeans.fit(reduced_df)
labels=model.predict(reduced_df)
reduced_df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Number of datapoints in cluster 1 (K Means):", count1)
print("Number of datapoints in cluster 2 (K Means):", count2)
print("Number of datapoints in cluster 3 (K Means):", count3)
print("Number of datapoints in cluster 4 (K Means):", count4)
for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1])
plt.legend(u_labels)
plt.show()

# Create a data frame for centroids

df = df1[['goal', 'staff_pick', 'create_to_launch_days', 'blurb_len_clean', 'launched_at_yr', 'created_at_yr', 'deadline_yr']]

scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
standardized_x=pd.DataFrame(standardized_x,columns=df.columns)
df=standardized_x
kmeans=KMeans(n_clusters=4)
model=kmeans.fit(df)
labels=model.predict(df)
df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Number of datapoints in cluster 1 (K Means):", count1)
print("Number of datapoints in cluster 2 (K Means):", count2)
print("Number of datapoints in cluster 3 (K Means):", count3)
print("Number of datapoints in cluster 4 (K Means):", count4)

import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.express as px
from pandas.plotting import *
centroids = pd.DataFrame(kmeans.cluster_centers_)
fig = px.parallel_coordinates(centroids,labels=df.columns,color=u_labels)
fig.show()



#########################################################
############### Kickstarter Working Code ################
#########################################################

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

############### ORIGINAL DATASET ###############

# Data import
df_og = pd.read_excel(path+"Kickstarter.xlsx")

# Re-arranging order to bring state to the 0th position
df_og = df_og[['state', 'project_id', 'name', 'goal', 'pledged', 'disable_communication', 'country', 'currency', 'deadline',
 'state_changed_at', 'created_at', 'launched_at', 'staff_pick', 'backers_count', 'static_usd_rate', 'usd_pledged', 'category',
 'spotlight', 'name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean', 'deadline_weekday', 'state_changed_at_weekday',
 'created_at_weekday', 'launched_at_weekday', 'deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr',
 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 'created_at_month',
 'created_at_day', 'created_at_yr', 'created_at_hr', 'launched_at_month', 'launched_at_day', 'launched_at_yr',
 'launched_at_hr', 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days']]

print(df_og.columns)
print("\nNumber of rows & columns:", df_og.shape)
df_og.head(5)

############### DATA CLEANING ###############

df_clean = df_og.copy()

# Cleaning state variable
print("State variable original counts:")
print(df_clean['state'].value_counts())

df_clean = df_clean[(df_clean["state"] == "failed") | (df_clean["state"] == "successful")]  # only retaining rows where state is successful or failed
df_clean["state"] = df_clean["state"].map({'successful': 1, 'failed': 0}).astype(int)  # mapping successful as 1, failed as 0
print("\nState variable updated counts:")
print(df_clean['state'].value_counts())

# Dropping fields which are decided after the launch date of the project
df_clean.drop(['usd_pledged', 'backers_count', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr',
          'state_changed_at_hr', 'state_changed_at_weekday'], axis=1, inplace=True)

# Dropping all the fields which will likely not make a difference to the project's success or failure
# Retaining blurb_len_clean instead of blurb_len, retaining usd_pledged instead of pledged
df_clean.drop(['name', 'name_len', 'name_len_clean', 'blurb_len', 'pledged', 'currency', 'static_usd_rate'], axis=1, inplace=True)

# Dropping the date fields since they are repetitive, will remove 'project_id' towards the end
df_clean.drop(['deadline', 'launched_at', 'created_at', 'state_changed_at'], axis=1, inplace=True)

# Fixing NULLs
print("\nPrinting NULL Stats:")
print(df_clean.isnull().sum()) # We note 1471 NULLS in category, and 10299 NULLS in launch_to_state_change_days
df_clean.drop(['launch_to_state_change_days'], axis=1, inplace=True)  #We dropped this column since 70% of its values were NULL

df_clean = df_clean.dropna()  #<10% of the category field was NULL, hence dropping NULLS
print("\nPrinting NULL Stats:")
print(df_clean.isnull().sum())

# Checking correlation
df_clean.corr()
df_clean.drop(['spotlight'], axis=1, inplace=True) # Dropping spotlight since all highly correlated with state

print("\nValue Counts for disable_communication:")
print(df_clean['disable_communication'].value_counts())  # Noticed NaN for disable_communication in corr matrix, hence exploring value_counts
df_clean.drop(['disable_communication'], axis=1, inplace=True) #Dropping disable_communication since all values are same (i.e., False)

print(df_clean.columns)
print("\n",df_clean.shape)

############### HANDLING CATEGORICAL FIELDS ###############
# Identifying categorical columns
cols = df_clean.columns
num_cols = df_clean._get_numeric_data().columns
print("Categorical columns:")
print(list(set(cols) - set(num_cols)))

# Creating dummy variables
dummy_category = pd.get_dummies(df_clean['category'], prefix="category")
dummy_created_at_weekday = pd.get_dummies(df_clean['created_at_weekday'], prefix="created_at_weekday")
dummy_deadline_weekday = pd.get_dummies(df_clean['deadline_weekday'], prefix="deadline_weekday")
dummy_country = pd.get_dummies(df_clean['country'], prefix="country")
dummy_launched_at_weekday = pd.get_dummies(df_clean['launched_at_weekday'], prefix="launched_at_weekday")

# Joining dummy fields into the main dataset
df1 = df_clean.copy()
df1 = df1.join(dummy_category)
df1 = df1.join(dummy_created_at_weekday)
df1 = df1.join(dummy_deadline_weekday)
df1 = df1.join(dummy_country)
df1 = df1.join(dummy_launched_at_weekday)

# Dropping the original categorical fields from the dataset
df1.drop(['category', 'created_at_weekday', 'deadline_weekday', 'country', 'launched_at_weekday'], axis=1, inplace=True)

# Dropping 'project_id'
proj_id = list(df1['project_id'])
df1.drop(['project_id'], axis=1, inplace=True)

df1.reset_index(drop=True, inplace=True)

print("\n",df1.shape)

df1.corr()

#############################################################
############### Classification Model Building ###############
#############################################################

## Creating GBT function

def gbt_thresh(X):
    y = df1["state"]

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=5)
    from sklearn.ensemble import GradientBoostingClassifier
    gbt = GradientBoostingClassifier() 
    model = gbt.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_test_pred)

    # Calculate the accuracy score
    from sklearn import metrics
    y_test_pred = model.predict(X_test)
    r1 = metrics.accuracy_score(y_test, y_test_pred)

    # Print the confusion matrix
    metrics.confusion_matrix(y_test, y_test_pred)
    # Calculate the Precision/Recall
    from sklearn import metrics
    r2 = metrics.precision_score(y_test, y_test_pred)
    r3 = metrics.recall_score(y_test, y_test_pred)
    # Calculate the F1 score
    r4 = metrics.f1_score(y_test, y_test_pred)
    
    return r1, r2, r3, r4


## Creating Logistic Regression function

def logistic_reg(X):
    y = df1["state"]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    from sklearn.linear_model import LogisticRegression
    lr2 = LogisticRegression()
    model = lr2.fit(X_train,y_train)
    
    y_test_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_test_pred)

    # Calculate the accuracy score
    from sklearn import metrics
    y_test_pred = model.predict(X_test)
    r5 = metrics.accuracy_score(y_test, y_test_pred)

    # Print the confusion matrix
    metrics.confusion_matrix(y_test, y_test_pred)
    # Calculate the Precision/Recall
    from sklearn import metrics
    r6 = metrics.precision_score(y_test, y_test_pred)
    r7 = metrics.recall_score(y_test, y_test_pred)
    # Calculate the F1 score
    r8 = metrics.f1_score(y_test, y_test_pred)
    
    return r5, r6, r7, r8

## KNN

def knn_algo(X):
    y = df1["state"]
    
    from sklearn.preprocessing import StandardScaler
    standardizer = StandardScaler()
    X_std = standardizer.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.3, random_state = 5)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    
    a = []
    for i in range (1,20):
        knn3 = KNeighborsClassifier(n_neighbors=i)
        model3 = knn3.fit(X_train,y_train)
        y_test_pred = model3.predict(X_test)
        a.append(accuracy_score(y_test, y_test_pred))
        
    return max(a)

## Random Forest

def rf_algo(X):
    y = df1["state"]
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=5)
    from sklearn.ensemble import RandomForestClassifier
    randomforest = RandomForestClassifier()
    model = randomforest.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    a = accuracy_score(y_test, y_test_pred)
    
    return a

## CART

def cart(X):
    y = df1["state"]
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=5)
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    
    a = []
    
    for i in range(2,21):
        decisiontree = DecisionTreeClassifier(max_depth=i)
        model = decisiontree.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        a.append(accuracy_score(y_test, y_test_pred))
        
    return max(a)


## Running GBT, Logistic Regression, KNN, Random Forest, CART before feature selection
X = df1.iloc[:,1:]

r1, r2, r3, r4 = gbt_thresh(X)
r5, r6, r7, r8 = logistic_reg(X)
r9 = knn_algo(X)
r10 = rf_algo(X)
r11 = cart(X)

print("GBT Results:")
print("Accuracy:", r1)
print("F1:", r4)

print("\nLogistic Regression Results:")
print("Accuracy:", r5)
print("F1:", r8)

print("\nKNN Results:")
print("Accuracy:", r9)

print("\nRandom Forest Results:")
print("Accuracy:", r10)

print("\nCART:")
print("Accuracy:", r11)


############# Feature Selection ############# 

############# Recursive Feature Elimination for feature selection
X = df1.iloc[:,1:]
y = df1["state"]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
rfe = RFE(lr, n_features_to_select=1)

model = rfe.fit(X, y)
model.ranking_
rfe_pred = pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','ranking'])
rfe_pred.sort_values(by="ranking", inplace=True)
rfe_pred.reset_index(drop=True, inplace=True)
rfe_pred.head(3)

############# Running loop for finding the optimal threshold for RFE feature selection

temp_a = rfe_pred['predictor']
temp_b = []
thresh = []
accuracy_score = []
f1_score = []

for i in range(len(rfe_pred)):
    temp_b.append(temp_a[i])
    
    X = df1[temp_b]
    r1, r2, r3, r4 = gbt_thresh(X)
    thresh.append(i)
    accuracy_score.append(r1)
    f1_score.append(r4)
    
df_gbt_RFE_thresh = pd.DataFrame(list(zip(thresh, accuracy_score, f1_score)), columns=['Threshold', 'Accuracy Score', 'F1 Score'])
df_gbt_RFE_thresh

print("Highest Accuracy Score values:")
print(df_gbt_RFE_thresh[df_gbt_RFE_thresh["Accuracy Score"]==max(df_gbt_RFE_thresh["Accuracy Score"])].sort_values(by=['Threshold']))

print("\nHighest F1 Score values:")
print(df_gbt_RFE_thresh[df_gbt_RFE_thresh["F1 Score"]==max(df_gbt_RFE_thresh["F1 Score"])].sort_values(by=['Threshold']))

############# LASSO for feature selection

X = df1.iloc[:,1:]
y = df1["state"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

from sklearn.linear_model import Lasso

temp_a = []
temp_b = [x / 1000 for x in range(10, 50, 1)]
temp_c = []

for i in temp_b:
    model = Lasso(alpha=i)
    model.fit(X_std,y)
    temp_a = pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])
    temp_a = list(temp_a[temp_a['coefficient']!=0]['predictor'])
    temp_c.append(temp_a)

temp_d = [list(x) for x in set(tuple(x) for x in temp_c)]

for i in temp_d:
    temp_e = df1[i]
    r1, r2, r3, r4 = gbt_thresh(temp_e)
    accuracy_score.append(r1)
    f1_score.append(r4)
    
df_gbt_Lasso_thresh = pd.DataFrame(list(zip(accuracy_score, f1_score)), columns=['Accuracy Score', 'F1 Score'])
df_gbt_Lasso_thresh

print("Highest Accuracy Score values:")
print(df_gbt_Lasso_thresh[df_gbt_Lasso_thresh["Accuracy Score"]==max(df_gbt_Lasso_thresh["Accuracy Score"])])

print("\nHighest F1 Score values:")
print(df_gbt_Lasso_thresh[df_gbt_Lasso_thresh["F1 Score"]==max(df_gbt_Lasso_thresh["F1 Score"])])

############# Random Forest for feature selection
X = df1.iloc[:,1:]
y = df1["state"]

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X, y)
model.feature_importances_

df_impt1 = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','feature importance'])
df_impt1.sort_values(by=['feature importance'], ascending=False, inplace=True)
df_impt1.head(5)

############# Running loop for finding the optimal threshold for random forest feature selection

temp_a = list(df_impt1['feature importance'])
thresh = []
accuracy_score = []
f1_score = []

for i in temp_a:
    temp_b = df_impt1[df_impt1['feature importance']>=i]
    temp_c = list(temp_b['predictor'])
    
    X = df1[temp_c]
    r1, r2, r3, r4 = gbt_thresh(X)
    thresh.append(i)
    accuracy_score.append(r1)
    f1_score.append(r4)
    
df_gbt_RF_thresh = pd.DataFrame(list(zip(thresh, accuracy_score, f1_score)), columns=['Threshold', 'Accuracy Score', 'F1 Score'])
df_gbt_RF_thresh

print("Highest Accuracy Score values:")
print(df_gbt_RF_thresh[df_gbt_RF_thresh["Accuracy Score"]==max(df_gbt_RF_thresh["Accuracy Score"])].sort_values(by=['Threshold']))

print("\nHighest F1 Score values:")
print(df_gbt_RF_thresh[df_gbt_RF_thresh["F1 Score"]==max(df_gbt_RF_thresh["F1 Score"])].sort_values(by=['Threshold']))

############# Grid Search

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
parameters = {'max_depth':[i for i in range(1,11)], 'n_estimators':[50, 100],
              'learning_rate':[0.01, 0.1, 1], 'max_features': ['log2', 'sqrt']}
grid = GridSearchCV(estimator=model, param_grid=parameters , cv=None, verbose=True)
grid.fit(X_std,y)

print(grid.best_params_)
print(grid.best_score_)

# Parameters by Grid Search: learning_rate= 0.1, max_depth= 4, max_features= 'log2', n_estimators= 100
# However, the accuracy of the model with these results was 0.77 and F1 score was 0.61
# Since this is worse-performing than the final model, we do not use Grid Search to evaluate results

############### Running GBT, Logistic Regression, KNN, Random Forest, CART after Random Forest feature selection (for optimal threshold = 0.0012)

df_impt_filter1 = df_impt1[df_impt1['feature importance']>=0.0012]
a = list(df_impt_filter1['predictor'])

X = df1[a]
y = df1["state"]

r1, r2, r3, r4 = gbt_thresh(X)
r5, r6, r7, r8 = logistic_reg(X)
r9 = knn_algo(X)
r10 = rf_algo(X)
r11 = cart(X)

print("Selected", len(a), "features:")
print(a)

print("\nGBT Results:")
print("Accuracy:", r1)
print("F1:", r4)

print("\nLogistic Regression Results:")
print("Accuracy:", r5)
print("F1:", r8)

print("\nKNN Results:")
print("Accuracy:", r9)

print("\nRandom Forest Results:")
print("Accuracy:", r10)

print("\nCART:")
print("Accuracy:", r11)

######## THANKS! ########
