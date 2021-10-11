#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ast


# In[2]:


import os
os.getcwd()
os.chdir('C:\\Users\\Aman Khan\\OneDrive\\Desktop\\Artificial Intelligence Projects\\Movie Recommender Systems')
os.getcwd()


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.shape


# In[5]:


credits.shape


# In[6]:


movies = movies.merge(credits,on='title')
movies.shape


# In[7]:


movies.head(1)


# In[8]:


movies.info()


# In[9]:


'''
id
title
overview
genres
keywords
cast
crew
'''

movies = movies[['id','title','overview','genres','keywords','cast','crew']]
movies.info()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.head(5)


# In[14]:


movies['genres'].values


# In[15]:


def convert(obj):
    l = []    
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[16]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[17]:


movies['genres'] = movies['genres'].apply(convert)


# In[18]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[19]:


movies['genres'].head(3)


# In[20]:


movies.head(1)


# In[21]:


def convert3(obj):
    l = []
    c = 0
    for i in ast.literal_eval(obj):
        if c!=3:
            l.append(i['name'])
            c+=1
        else:
            break
    return l


# In[22]:


movies['cast'] = movies['cast'].apply(convert3)


# In[23]:


movies.head(1)


# In[24]:


movies['crew'].head(1).values


# In[25]:


import ast
def fetch_director(obj):
    l = []
    
    for i in ast.literal_eval(obj):
        
        if i['job']=='Director':
            l.append(i['name'])
            break
            
    return l


# In[26]:


movies['crew']= movies['crew'].apply(fetch_director)


# In[27]:


movies.head(10)


# In[ ]:





# In[28]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
    


# In[29]:


movies.head()


# In[30]:


'''def sname(fname):
    f2name = ''.join('')
    l = fname.split()
    short = []
    for i in l:
        short.append(i[0:2])
    return short'''


# In[31]:


movies['crew'][0]


# In[32]:


'''movies['crew'].apply(sname)'''


# In[33]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[34]:


movies.head()


# In[35]:


movies['overview'].head(1).values


# In[36]:


movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[37]:


movies['tags'].head(1).values


# In[38]:


new_df = movies[['id','title','tags']]


# In[39]:


new_df


# In[40]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[41]:


new_df['tags'][0]


# In[42]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[43]:


import nltk


# In[44]:


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# In[45]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[46]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[47]:


new_df.head()


# In[48]:


new_df['tags'][0]


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 5000,stop_words = 'english')


# In[50]:


vectors  = cv.fit_transform(new_df['tags']).toarray()


# In[51]:


vectors


# In[52]:


vectors[0]


# In[53]:


cv.get_feature_names()


# In[54]:


#stemming
stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[55]:


from sklearn.metrics.pairwise import cosine_similarity


# In[56]:


cosine_similarity(vectors)


# In[57]:


similarity = cosine_similarity(vectors)


# In[58]:


sorted(list(enumerate(similarity[0])),reverse = True , key = lambda x:x[1])[1:6]


# In[59]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[60]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    dist = similarity[movie_index]
    movie_list = sorted(list(enumerate(dist)),reverse = True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        


# In[64]:


movie = input()
recommend(movie)


# In[ ]:


new_df.iloc[1216].title


# In[ ]:





# In[ ]:




