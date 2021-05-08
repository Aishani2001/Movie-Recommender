'''

                            Online Python Compiler.
                Code, Compile, Run and Debug python program online.
Write your code in this editor and press "Run" button to execute it.

'''
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import flask
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv('movies.csv')
app = flask.Flask(__name__, template_folder='templates')
count = CountVectorizer()
count_matrix = count.fit_transform(movies['All_Words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
indices = pd.Series(movies.index)
all_titles = [movies['Title'][i] for i in range(len(movies['Title']))]

def recommendations(Title, cosine_sim = cosine_sim):
    
    # initializing the empty list of recommended movies
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == Title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(movies.index)[i])
        
    return recommended_movies

def get_recommendations(Title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[Title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = movies['Title'].iloc[movie_indices]
    dat = movies['All_Words'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title','All_Words'])
    return_df['Title'] = tit
    return_df['All_Words'] = dat
    return return_df   
    
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('mainpage.html'))
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
#        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        #if m_name in all_titles:
            #return(flask.render_template('negative.html',name=m_name))
        
        result_final = recommendations(m_name)
        names = []
        names.append(result_final)
        #for i in range(len(result_final)):
            #  names.append(result_final.iloc[i])
            #   dates.append(result_final.iloc[i])

        return flask.render_template('positive.html',movie_names=names,search_name=m_name)

if __name__ == '__main__':
    app.run()     
        
        
        
        
