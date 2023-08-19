import streamlit as st
import pickle
import pandas as pd
import requests
from IPython.display import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


movie_list=pickle.load(open('movies_dict.pkl','rb'))
movies=pd.DataFrame(movie_list)
# obj=pickle.load(open('sim_obj.pkl','rb'))
CV=CountVectorizer(max_features=5000)
vectors=CV.fit_transform(movies['tags']).toarray()
obj=cosine_similarity(vectors)


def get_Recommondation(movie_name):
    movieIndex=movies[movies['title'] == movie_name].index[0]
    l=list(enumerate(obj[movieIndex]))
    sorted_L=sorted(l,key=lambda x:x[1],reverse=True)
    image_urls=[]
    movie_name=[]
    for i in range(1,7):
        j=sorted_L[i][0]
        movie_name.append(movies['title'][j])
        image_urls.append(getPoster(movies['id'][j]))
    st.image(image_urls,caption=movie_name,width=150)


def getPoster(movie_Id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=526aeb919c5523f99c47f8d28998794a'.format(movie_Id))
    data=response.json()
    print("TITE :",data['original_title'])
    print("Image :",data['poster_path'])
    url='https://image.tmdb.org/t/p/w185{}'.format(data['poster_path'])
    # print(url)
    # st.image(url,caption=data['original_title'],use_column_width=False,width=100)
    return url


st.title('Movie Recommender System')
st.divider()

option = st.selectbox(
    'Choose a movie ',
    movies['title'])
st.write("You may like...")
# if st.button('Get Recommondations!'):
get_Recommondation(option)
