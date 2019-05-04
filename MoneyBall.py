import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import statsmodels.api as sm


#Reads in data file and cleans up data
df = pd.read_csv('/Users/michaelnamoff/Desktop/baseball.csv')
df['Year'] = df['Year'].astype(float)
df['Playoffs'] = df['Playoffs'].astype(float)
df['W'] = df['W'].astype(float)


#Billy Bean did these perdictions based in 2002 so we can only use data eariler than 2002
dataBefore2002 = (df[df.Year < 2002])
#Creates a plot that shows the number of wins each team has. Yellow dots shows all the teams that made the playoffs
y = dataBefore2002['Team']
x = dataBefore2002['W']
plt.scatter(x, y, c=(dataBefore2002['Playoffs']))
plt.title('Number of Wins to Make The Playoffs')
plt.xlabel("Wins")
plt.ylabel("Team")
line = [95]
plt.axvline(x=line, c='r')
plt.show()


#In order to find how many run it takes to win we need to create a Run Differential column in our data set
rd = (dataBefore2002['RS'] - dataBefore2002['RA'])
dataBefore2002['RD'] = rd


#Shows the relationship between Run Differential and Wins
wins = dataBefore2002['W']
runD = dataBefore2002['RD']
slope, intercept, r_value, p_value, std_err = stats.linregress(runD, wins)
print ('This is slope:', slope, 'The intercept is:', intercept, 'The R value is:', r_value)


def predict(x):
    return slope * x + intercept
fitLine = predict(runD)
plt.scatter(runD, wins)
plt.title('Run Differential VS Number of Wins')
plt.xlabel('Run Differential')
plt.ylabel('# of Wins')
plt.plot(runD, fitLine, c='r')
print(plt.show())


w = dataBefore2002[['SLG', 'BA', 'OBP']]
z = dataBefore2002['RS']
est = sm.OLS(z, w).fit()
print(est.summary())
v = dataBefore2002 [['SLG']]
k =dataBefore2002 [['BA']]
c = dataBefore2002 [['OBP']]
w = dataBefore2002 [['RS']]



plt.scatter(w,v, c='r', label = 'SLG')
plt.scatter(w,k, c='g', label = 'BA')
plt.scatter(w,c, c='b', label = 'OBP')
plt.legend()
plt.title('SLG|BA|OBP VS Runs Scored')
plt.xlabel('Runs Scored')
plt.ylabel('%')
plt.show()















