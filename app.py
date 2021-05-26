import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__, template_folder='templates')
movies = pd.read_csv('movies.csv')
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
    idx = movies[movies['Title']==Title].index[0]  #indices[indices == Title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(all_titles[i])
        
    return recommended_movies

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
        
        result_final = recommendations(m_name,cosine_sim=cosine_sim)
        names = []
        names.extend(result_final)
        #for i in range(len(result_final)):
            #  names.append(result_final.iloc[i])
            #   dates.append(result_final.iloc[i])

        return flask.render_template('positive.html',movie_names=names,search_name=m_name)

if __name__ == '__main__':
    app.debug=True
    app.run()     
