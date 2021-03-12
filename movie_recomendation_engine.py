import pandas as pd 
import numpy as np
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
import re

CURRENT_USER = 672
def read_data():
	# imdb movie list with user ratings
	...


def init(path):
	df = pd.DataFrame()
	movies = get_movies(path)
	ratings = get_ratings(path)
	links = get_links(path)

	n_users = ratings['userId'].unique().shape[0]
	n_items = ratings['movieId'].unique().shape[0]
	print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

	df = pd.merge(movies, ratings, on="movieId")
	full_df = pd.merge(df, links, on="movieId")

	dates_list = []
	for s in full_df.title:
		date = s[s.find("(")+1:s.find(")")]
		try:
			date = int(date)
		except:
			date = 0
		dates_list.append(date)
	full_df['Date'] = dates_list


	currentuser = CURRENT_USER
	ratings_pivot_table  = full_df.pivot_table('rating', index = 'movieId', columns = 'userId')
	user_ratings = ratings_pivot_table[currentuser]
	user_correlation = ratings_pivot_table.corrwith(user_ratings)

	full_df_reduced = full_df[user_ratings[full_df.movieId].isnull().values & (full_df.userId != currentuser) & (full_df.Date > 1900)]
	full_df_reduced['similarity'] = full_df_reduced['userId'].map(user_correlation.get)
	full_df_reduced['sim_rating'] = full_df_reduced['similarity'] * full_df_reduced.Date


	recomend = full_df_reduced.groupby('movieId').apply(lambda s: s['sim_rating'].sum() / s['similarity'].sum())
	sortedrec = recomend.sort_values(ascending = False)
	
	recitems = sortedrec.index[:50]
	recomended_titles = []
	for anID in recitems:
		output = full_df[full_df['movieId'] == anID]
		title = output['title'].to_string() 
		x = re.sub(r'\([^)]*\)', '', title).split('\n')[0] 
		y = x.split(' ', 1)[-1]  
		recomended_titles.append(y)
	df = pd.DataFrame()
	df['Movies'] = recomended_titles
	
	return df['Movies'] 



def get_user_ratings(path):
	df = get_ratings(path)
	userdata = df[df.userId == CURRENT_USER]
	user_movieids = userdata.movieId

	allmovies = get_movies(path)
	allmovies = allmovies.set_index(['movieId'])

	user_rated_movies = []
	for item in user_movieids:
		user_rated_movies.append(allmovies['title'].ix[item])

	return user_rated_movies