import numpy as np
import pandas as pd
from scipy import stats

movie_df=pd.read_csv('movieReplicationSet.csv')

popularMovie_df=[]
lesspopularMovie_df=[]
df=movie_df.iloc[:, 0:400]
med=df.isna().sum().median()
med1=df.isna().sum()
for i in range(0,400):
  x=med1[i]
  if(x<med):
    popularMovie_df+=[movie_df.iloc[:, i]]
  else:
    lesspopularMovie_df+=[movie_df.iloc[:, i]]
popularMovie_df=pd.DataFrame(popularMovie_df)
lesspopularMovie_df=pd.DataFrame(lesspopularMovie_df)
popularMovie_df=popularMovie_df.T
lesspopularMovie_df=lesspopularMovie_df.T

M1=np.array(popularMovie_df.median())
M2=np.array(lesspopularMovie_df.median())

u,p = stats.mannwhitneyu(M1,M2)
# =============================================================================
# c=0
# p_value=[]
# for i in range(0,200):
#     x=(popularMovie_df.iloc[:, i])
#     x = x[~np.isnan(x)]
# # =============================================================================
# #     y=(lesspopularMovie_df.dropna().values.flatten())
# # =============================================================================
#     for j in range(0,200):
#         y=(lesspopularMovie_df.iloc[:, j])
#         y = y[~np.isnan(y)]
#         u,p = stats.mannwhitneyu(x,y)
#         if(p<0.005):
#             p_value+=[p]
#             c=c+1
# =============================================================================

# =============================================================================
# M1=np.array(popularMovie_df.values.flatten())
# M1 = M1[~np.isnan(M1)]
# M2=np.array(lesspopularMovie_df.values.flatten())
# M2 = M2[~np.isnan(M2)]
# 
# u,p = stats.mannwhitneyu(M1,M2)
# t1,p1 = stats.ttest_ind(M1,M2)
# =============================================================================

#%% Q2

import re
newMovie=[]
oldMovie=[]
releaseDate=[]
df= df.rename(columns={'Rambo: First Blood Part II': 'Rambo: First Blood Part II (1985)'})
column_names=(df.columns)
for i in range(0,400):
  x=column_names[i]
  match=re.match(r'.*([1-3][0-9]{3})', x)
  releaseDate+=[match.group(1)]
releaseDate.sort()
releaseDate= np.array(releaseDate, dtype=np.int)
med=np.median(releaseDate)
cnt=0
for i in range(0,400):
    x=column_names[i]
    match=re.match(r'.*([1-3][0-9]{3})', x)
    if match is not None:
        # Then it found a match!
        if int(match.group(1)) <= med and cnt<200:
            cnt=cnt+1
            oldMovie+=[df.iloc[:,i]]
        else:
            newMovie+=[df.iloc[:,i]]

newMovie_df=pd.DataFrame(newMovie)
oldMovie_df=pd.DataFrame(oldMovie)
newMovie_df=newMovie_df.T
oldMovie_df=oldMovie_df.T

M1=np.array(newMovie_df.median())
M2=np.array(oldMovie_df.median())

u,p = stats.mannwhitneyu(M1,M2)
# =============================================================================
# M1=np.array(newMovie_df.values.flatten())
# M1 = M1[~np.isnan(M1)]
# M2=np.array(oldMovie_df.values.flatten())
# M2 = M2[~np.isnan(M2)]
# 
# u,p = stats.mannwhitneyu(M1,M2)
# t1,p1 = stats.ttest_ind(M1,M2)
# =============================================================================

#%% Q3

# =============================================================================
# df1=[]
# df1=movie_df[['Shrek (2001)', 'Gender identity (1 = female; 2 = male; 3 = self-described)']]
# print(df1.groupby('Gender identity (1 = female; 2 = male; 3 = self-described)').mean())
# =============================================================================
femaleratedMovie=[]
maleRatedMovie=[]
for i in range(0,1097): 
    x=movie_df.iloc[i,474]
    if(x==1):
        femaleratedMovie+=[df.iloc[i,87]]
    elif(x==2):
        maleRatedMovie+=[df.iloc[i,87]]

femaleratedMovie_df=pd.DataFrame(femaleratedMovie)
maleRatedMovie_df=pd.DataFrame(maleRatedMovie)

M1=np.array(femaleratedMovie_df)
M1= M1[~np.isnan(M1)]
M2=np.array(maleRatedMovie_df)
M2= M2[~np.isnan(M2)]

k,p=stats.kstest(M1,M2)
# =============================================================================
# u,p = stats.mannwhitneyu(M1,M2)
# t1,p1 = stats.ttest_ind(M1,M2)
# =============================================================================

#%% Q4

import math
femaleratedMovie=[]
maleRatedMovie=[]
for i in range(0,1097):
    x=movie_df.iloc[i,474]
    if(x==1):
        femaleratedMovie+=[df.iloc[i,:]]
    elif(x==2):
        maleRatedMovie+=[df.iloc[i,:]]

femaleratedMovie_df=pd.DataFrame(femaleratedMovie)
maleRatedMovie_df=pd.DataFrame(maleRatedMovie)

cnt=0
for i in range(0,400):
    M1=femaleratedMovie_df.iloc[:,i]
    M1= M1[~np.isnan(M1)]
    M2=maleRatedMovie_df.iloc[:,i]
    M2= M2[~np.isnan(M2)]
    k,p=stats.kstest(M1,M2)
    if(p<0.005):
        cnt=cnt+1
    
proportion=(cnt/400)*100

#%% Q5

onlychildMovie=[]
siblingMovie=[]
for i in range(0,1097):
    x=movie_df.iloc[i,475]
    if(x==1):
        onlychildMovie+=[df.iloc[i,220]]
    elif(x==0):
        siblingMovie+=[df.iloc[i,220]]

onlychildMovie_df=pd.DataFrame(onlychildMovie)
siblingMovie_df=pd.DataFrame(siblingMovie)


M1=np.array(onlychildMovie_df.values.flatten())
M1= M1[~np.isnan(M1)]
M2=np.array(siblingMovie_df.values.flatten())
M2= M2[~np.isnan(M2)]

k,p=stats.kstest(M1,M2)

#%% Q6

onlychildMovie=[]
siblingMovie=[]
for i in range(0,1097):
    x=movie_df.iloc[i,475]
    if(x==1):
        onlychildMovie+=[df.iloc[i,:]]
    elif(x==0):
        siblingMovie+=[df.iloc[i,:]]

onlychildMovie_df=pd.DataFrame(onlychildMovie)
siblingMovie_df=pd.DataFrame(siblingMovie)

cnt=0
for i in range(0,400):
    M1=onlychildMovie_df.iloc[:,i]
    M2=siblingMovie_df.iloc[:,i]
    M1= M1[~np.isnan(M1)]
    M2= M2[~np.isnan(M2)]
    k,p=stats.kstest(M1,M2)
    if(p<0.005):
        cnt=cnt+1
    
proportion=(cnt/400)*100

#%% Q7

aloneMovie=[]
togetherMovie=[]
for i in range(0,1097):
    x=movie_df.iloc[i,476]
    if(x==1):
        aloneMovie+=[df.iloc[i,357]]
    elif(x==0):
        togetherMovie+=[df.iloc[i,357]]

aloneMovie_df=pd.DataFrame(aloneMovie)
togetherMovie_df=pd.DataFrame(togetherMovie)


M1=np.array(aloneMovie_df.values.flatten())
M1= M1[~np.isnan(M1)]
M2=np.array(togetherMovie_df.values.flatten())
M2= M2[~np.isnan(M2)]

k,p=stats.kstest(M1,M2)

#%% Q8


aloneMovie=[]
togetherMovie=[]
for i in range(0,1097):
    x=movie_df.iloc[i,476]
    if(x==1):
        aloneMovie+=[df.iloc[i,:]]
    elif(x==0):
        togetherMovie+=[df.iloc[i,:]]

aloneMovie_df=pd.DataFrame(aloneMovie)
togetherMovie_df=pd.DataFrame(togetherMovie)

cnt=0
for i in range(0,400):
    M1=aloneMovie_df.iloc[:,i]
    M2=togetherMovie_df.iloc[:,i]
    M1= M1[~np.isnan(M1)]
    M2= M2[~np.isnan(M2)]
    k,p=stats.kstest(M1,M2)
    if(p<0.005):
        cnt=cnt+1
    
proportion=(cnt/400)*100


#%% Q9

M1=np.array(df.iloc[:,285])
M2=np.array(df.iloc[:,138])
M1= M1[~np.isnan(M1)]
M2= M2[~np.isnan(M2)]
# =============================================================================
# temp = np.array([np.isnan(M1),np.isnan(M2)],dtype=bool)
# temp2 = temp*1 # convert boolean to int
# temp2 = sum(temp2) # take sum of each participant
# missingData = np.where(temp2>0) # find participants with missing data
# M1 = np.delete(M1,missingData) # delete missing data from array
# M2 = np.delete(M2,missingData) # delete missing data from array
# =============================================================================

k,p=stats.kstest(M1,M2)

if p<0.005:
    print("TRUE")
else:
    print("False")
    
#%% Q10
 
p_value=[]
movies=['Star Wars', 'Harry Potter', 'The Matrix', 'Indiana Jones', 'Jurassic Park', 'Pirates of the Caribbean', 'Toy Story', 'Batman']
for movie in movies:
    cnt=0
    franchiseMovies=[]
    for i in range(0,400):
        x=column_names[i]
        if movie in x:
            franchiseMovies+=[df.iloc[:,i]]
            
    franchiseMovies_df=pd.DataFrame(franchiseMovies)
    franchiseMovies_df=franchiseMovies_df.T
    M=np.array(franchiseMovies_df)
    M1=np.hsplit(M,franchiseMovies_df.shape[1])
    for i in range(0,len(M1)):
        M1[i]=M1[i][~np.isnan(M1[i])]
    
    if len(M1)==2:
        h,p = stats.kruskal(M1[0],M1[1])
        p_value+=[p]
    elif len(M1)==3:
        h,p = stats.kruskal(M1[0],M1[1],M1[2])
        p_value+=[p]
    elif len(M1)==4:
        h,p = stats.kruskal(M1[0],M1[1],M1[2],M1[3])
        p_value+=[p]
    elif len(M1)==5:
        h,p = stats.kruskal(M1[0],M1[1],M1[2],M1[3],M1[4])
        p_value+=[p]
    elif len(M1)==6:
        h,p = stats.kruskal(M1[0],M1[1],M1[2],M1[3],M1[4],M1[5])
        p_value+=[p]
    
for i in range(0,len(p_value)):
    if(p_value[i]<0.005):
        print("Movie", movies[i], "is inconsistent")
    else:
        print("Movie", movies[i], "is consistent")
# =============================================================================
#     M1=np.array(franchiseMovies_df.iloc[:,0])
#     M1= M1[~np.isnan(M1)]
#     for j in range(0,franchiseMovies_df.shape[1]):
#         M2=np.array(franchiseMovies_df.iloc[:,j])
#         M2= M2[~np.isnan(M2)]
#         u,p = stats.mannwhitneyu(M1,M2)
#         t1,p1 = stats.ttest_ind(M1,M2)
#         print("P value:", p1)
#         if p1<0.005:
#             cnt=cnt+1
#     if cnt>0:
#         print("Incositent movie")
#     else:
#         print("Consistent movie")
# =============================================================================
    
#%% Q11

#imdb Package doesn't run on Spyder, trying running the code in Colab/Jupyter,
#Note: It takes around 20 mins to find genres of all the movie.
import imdb
ia = imdb.IMDb()
genres=[]
s=""
for i in range(0,400):
  name=column_names[i]
  search = ia.search_movie(name)
  if(len(search)>1):
    id = search[0].movieID
    movie = ia.get_movie(id)
    if str(movie) in name:
      for genre in movie['genres']:
        genres+=[genre]
        break
    else:
      genres+=["No Genre"]
  else:
    genres+=["No Genre"]
genres

action_movie=[]
for i in range(0,400):
  x=genres[i]
  if x=="Action":
    action_movie+=[df.iloc[:,i]]
action_movie_df=pd.DataFrame(action_movie)
action_movie_df=action_movie_df.T
femaleratedMovie=[]
maleRatedMovie=[]
for i in range(0,1097):
    x=movie_df.iloc[i,474]
    if(x==1):
        femaleratedMovie+=[action_movie_df.iloc[i,:]]
    elif(x==2):
        maleRatedMovie+=[df.iloc[i,:]]

femaleratedMovie_df=pd.DataFrame(femaleratedMovie)
maleRatedMovie_df=pd.DataFrame(maleRatedMovie)

M1=np.array(femaleratedMovie_df.median())
M2=np.array(maleRatedMovie_df.median())
u,p = stats.mannwhitneyu(M1,M2)