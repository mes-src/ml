import pandas as pd 
import numpy as np 
'''
in poisson method, change calculation of expected goals to epected goal diferential...better predictor of wins?
'''
def init(matchlist,oddsList):
	paths = ['data/SP1.csv']
	frameslist = []
	def construct_full_df(paths):
		for item in paths:
			df = pd.read_csv(item)

			dates = pd.to_datetime(df['Date'])
			df['Date'] = dates

			df = df.set_index('Date')
			frameslist.append(df)

	construct_full_df(paths)
	full_df = pd.concat(frameslist)
	print('Data Start Date: ' + str(full_df.index[0]))
	print('Data End Date: ' + str(full_df.index[len(full_df.index) -1]))
	# print(full_df.columns.values)

	teamNames = []

	for item in full_df['HomeTeam']:
		if item not in teamNames:
			teamNames.append(item)

	dfSelectedHome,dfSelectedAway = construct_selected_df(full_df, matchlist)

	probdf,goaldf = poisson(full_df, matchlist)
	homeExpValue,AwayExpValue = calculateExpectedValue(probdf, oddsList)
	return(probdf, homeExpValue, AwayExpValue, goaldf)



def construct_selected_df(full_df,matchlist):
	columnlist = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTR',
	'HTHG', 'HTR', 'HS','HST','B365H','B365D','B365A','PSH','PSD','PSA',
	'BbAHh', 'BbAvAHH','BbAvAHA']

	fullDfHome = full_df.loc[full_df['HomeTeam'] == matchlist[0]]
	fullDfAway = full_df.loc[full_df['HomeTeam'] == matchlist[1]]

	dfHome = fullDfHome[columnlist]
	dfAway = fullDfHome[columnlist]
	return (dfHome,dfAway)



def poisson(df,matchlist):
	import statsmodels.api as sm
	import statsmodels.formula.api as smf
	from scipy.stats import poisson,skellam

	homeTeam  =matchlist[0]
	awayTeam = matchlist[1]

	#POISSON REGRESSION USING A GENERALIZED LINEAR MODEL (GLM)
	goal_model_data = pd.concat([df[['HomeTeam','AwayTeam','FTHG']].assign(home=1).rename(
	            columns={'HomeTeam':'team', 'AwayTeam':'opponent','FTHG':'goals'}),
	           df[['AwayTeam','HomeTeam','FTAG']].assign(home=0).rename(
	            columns={'AwayTeam':'team', 'HomeTeam':'opponent','FTAG':'goals'})])

	poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
	                        family=sm.families.Poisson()).fit()
	summary = poisson_model.summary()
	#print(summary)

	#EXPECTED AVERAGE NUMBER OF GOALS SCORED BY EACH TEAM
	expected_goals_hometeam = poisson_model.predict(pd.DataFrame(data={'team': homeTeam, 'opponent': awayTeam,'home':0},index=[1]))
	expected_goals_awayteam = poisson_model.predict(pd.DataFrame(data={'team': awayTeam, 'opponent': homeTeam,'home':1},index=[1]))
	expectedGoalData = [float(expected_goals_hometeam.values), float(expected_goals_awayteam.values)]

	dfExpectedGoals = pd.DataFrame(expectedGoalData, columns = ['Goals'], index = ['Home', 'Away'])

	def simulate_match(foot_model, homeTeam, awayTeam, max_goals=4):
	    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
	                                                            'opponent': awayTeam,'home':1},
	                                                      index=[1])).values[0]
	    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
	                                                            'opponent': homeTeam,'home':0},
	                                                      index=[1])).values[0]
	    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
	    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

	simulation = simulate_match(poisson_model, homeTeam, awayTeam, max_goals=4)
	homewin = np.sum(np.tril(simulation, -1)) #.tril returns elements above the diagonal
	draw = np.sum(np.diag(simulation))
	awaywin = np.sum(np.triu(simulation, 1))

	goaldf = dfExpectedGoals

	dfScore = pd.DataFrame(simulation, columns = [i for i in range(5)], index = [i for i in range(5)])
	dfScore.index.name = 'Goals'

	dfProbability = pd.DataFrame([homewin, awaywin, draw], columns = ['Probability'], index = ['Home', 'Away', 'Draw'])
	dfProbability.index.name = 'Winner'
	print(dfProbability)

	return (dfProbability,goaldf)




def calculateExpectedValue(dfProbability,oddsList):
	import numpy as np
	loopcount = 0
	loopcount2 = 0
	def convertMLtoImpliedProb(oddsList,loopcount):
		impProbList = []
		for pair in oddsList: 
			for item in pair:
				if item > 0:#Positive
					ip = 100 / (item + 100)
				else:
					ip = -item / (-item + 100)
				impProbList.append(ip)
		loopcount += 1
		loopcount += 1
		return (impProbList,loopcount)

	homeWprob = np.round(dfProbability['Probability']['Home'], 3)
	homeAprob = np.round(dfProbability['Probability']['Away'], 3)

	prob,loopcount = convertMLtoImpliedProb(oddsList,loopcount)

	a = (homeWprob,  str(np.round(prob,2)))
	prob2,loopcount2 = convertMLtoImpliedProb(oddsList,loopcount2)
	b =  (homeAprob, np.round(prob2,2))
	return (a,b)



teamlist = [['Barcelona','Leganes'], ['Barcelona','Leganes'], ['Barcelona','Leganes'],['Barcelona','Leganes']]
oddsList = [[-575,2000],[2000,-1200],[-120,200]]










