# Movie-Recommender
**OVERVIEW**
Have you ever wondered where all the recommendations that Netflix, Amazon, Google give us, come from? We often rate products on the internet and all the preferences we express and data we share (explicitly or not), are used by recommender systems to generate, in fact, recommendations. We have built a content-based movie recommendation system using the IMDB dataset of top 250 rated movies. Content-based filters do not involve other users; they simply recommend movies on the basis of our past viewed content. Recommendation is generated according to Genres. Main Actors, Directors and Plot. We use cosine similarity to determine if the two movies are similar or not.

**FILE DESCRIPTION:** <br>
-train.csv := the training set <br>
=test.csv := the testing set <br>
-sample.csv := a sample submission in the correct format <br>

**REQUIREMENTS:** <br>
-Python3 must be installed <br>
-Numpy <br>
-Pandas <br>

**OTHER LIBRARIES USED:**
-Seaborn <br>
-Matplotlib <br>
-Scikit-Learn <br>

**ACKNOWLEDGMENTS**: We thank IMDB for providing us with the dataset.

**COSINE SIMILARITY:**
The dot product between two vectors is equal to the projection of one of them on the other. Therefore, the dot product between two identical vectors (i.e. with identical components) is equal to their squared module, while if the two are perpendicular (i.e. they do not share any directions), the dot product is zero. The dot product is important when defining the similarity, as it is directly connected to it. The definition of similarity between two vectors u and v is, in fact, the ratio between their dot product and the product of their magnitudes.
By applying the definition of similarity, this will be in fact equal to 1 if the two vectors are identical, and it will be 0 if the two are orthogonal. In other words, the similarity is a number bounded between 0 and 1 that tells us how much the two vectors are similar.

**MADE BY:**
-[Aishani Mitra] (https://github.com/Aishani2001) <br>
-[Shreya Singh] (https://github.com/ss0313)
