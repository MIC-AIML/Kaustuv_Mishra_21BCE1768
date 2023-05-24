#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System
# ### The movie recommendation system project aims to provide personalized movie recommendations to users based on their preferences and behavior. It utilizes collaborative filtering techniques to identify similar movies and make recommendations. Here's an overview of the project:
# 
# Data Collection: The project starts by obtaining the movie rating data, which typically includes information such as user IDs, movie IDs, ratings, and timestamps. In this case, the data is retrieved from a TSV file using pandas library.
# 
# Data Exploration: The collected data is explored to gain insights into the movies and ratings. This includes checking the data structure, merging with movie titles, calculating mean ratings and count of ratings for each movie, and visualizing the distribution of ratings.
# 
# User-Based Recommendations: Collaborative filtering is used to find movies that are similar to the ones a user has already rated highly. The project calculates the correlation between user ratings and the ratings of other movies. Specifically, it focuses on finding movies similar to "Star Wars (1977)" and "Liar Liar (1997)".
# 
# Item-Based Collaborative Filtering: Similar to user-based recommendations, item-based collaborative filtering identifies movies that are similar based on user ratings. This step helps in finding movies similar to a particular movie rather than a specific user's preferences.

# ---------------x---------------x---------------x---------------x---------------x---------------x---------------x---------------x---------------x---------------x---------------x---------------x------------

# The code begins by importing the required libraries: pandas for data manipulation, warnings to ignore warning messages, and matplotlib.pyplot and seaborn for data visualization.

# In[1]:


#Importing necessary libraries:
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
# This line of code ignores any warning messages that might be generated during the execution of the code.
warnings.filterwarnings("ignore") 


# ### Loading and exploring the data:
# Here, the code defines the column names for the dataset and the URL path from where the data is loaded. The pd.read_csv() function is used to read the data into a DataFrame (df).

# In[2]:


# Get the data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
path = 'https://media.geeksforgeeks.org/wp-content/uploads/file.tsv'
df = pd.read_csv(path, sep='\t', names=column_names)

# Check the head of the data
# This line of code displays the first few rows of the DataFrame df to provide a glimpse of the data.
df.head()


# ### Loading movie titles data:
# The code reads another dataset containing movie titles and their respective IDs from a CSV file using pd.read_csv(). The data is stored in the movie_titles DataFrame.

# In[3]:


# Check out all the movies and their respective IDs
movie_titles = pd.read_csv('https://media.geeksforgeeks.org/wp-content/uploads/Movie_Id_Titles.csv')

# Display the first few rows of movie_titles
movie_titles.head()


# ### Merging data and calculating mean rating/count:
# The code merges the two datasets (df and movie_titles) based on the common column 'item_id'. It then groups the data by movie title and calculates the mean rating and the count of ratings for each movie using the groupby() and agg() functions.

# In[4]:


data = pd.merge(df, movie_titles, on='item_id')

# Calculate mean rating and count of all movies
ratings = data.groupby('title')['rating'].agg(['mean', 'count'])
ratings.columns = ['mean_rating', 'num_of_ratings']


# ### Plotting the distribution of ratings:
# These lines of code use matplotlib.pyplot and seaborn to plot histograms showing the distribution of the number of ratings and the mean rating for movies, respectively.

# In[5]:


# Plotting the distribution of the number of ratings
plt.figure(figsize=(10, 4))
sns.histplot(ratings['num_of_ratings'], bins=70)
plt.xlabel('Number of Ratings')
plt.ylabel('Count')
plt.title('Distribution of Number of Ratings')

# Plotting the distribution of ratings
plt.figure(figsize=(10, 4))
sns.histplot(ratings['mean_rating'], bins=70)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Ratings')


# ### Creating a pivot table for user ratings:
# This code creates a pivot table (moviemat) where each row represents a user, each column represents a movie title, and the values represent the ratings given by users to movies.

# In[6]:


# Create a pivot table for user ratings
moviemat = data.pivot_table(index='user_id', columns='title', values='rating')


# ### Calculating correlations with specific movies:
# After calculating the correlation of user ratings with the movie "Star Wars (1977)", the code filters the results to consider only movies with a minimum of 100 ratings. Then, it sorts the movies based on the correlation in descending order and selects the top 10 movies with the highest correlation. The resulting DataFrame, corr_starwars, contains the top similar movies to "Star Wars (1977)" based on user ratings.

# In[7]:


# Calculate correlation with "Star Wars (1977)"
starwars_user_ratings = moviemat['Star Wars (1977)']
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars = corr_starwars.join(ratings['num_of_ratings'])
corr_starwars = corr_starwars[corr_starwars['num_of_ratings'] > 100]
corr_starwars = corr_starwars.sort_values('Correlation', ascending=False).head(10)


# The code for calculating correlations with the movie "Liar Liar (1997)" follows a similar structure and can be found in the original code:
# This code calculates the correlation of user ratings with the movie "Liar Liar (1997)" and applies similar filtering and sorting operations to obtain the top similar movies to "Liar Liar (1997)" based on user ratings.

# In[8]:


# Calculate correlation with "Liar Liar (1997)"
liarliar_user_ratings = moviemat['Liar Liar (1997)']
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num_of_ratings'])
corr_liarliar = corr_liarliar[corr_liarliar['num_of_ratings'] > 100]
corr_liarliar = corr_liarliar.sort_values('Correlation', ascending=False).head(10)


# ### Finally, the code prints the results:
# It displays the top similar movies to "Star Wars (1977)" and "Liar Liar (1997)" based on the correlation of user ratings.

# In[9]:


# Print the results
print("Movies similar to Star Wars (1977):")
print(corr_starwars)
print("\nMovies similar to Liar Liar (1997):")
print(corr_liarliar)


# Overall, the movie recommendation system project combines exploratory data analysis, collaborative filtering, and potentially machine learning to deliver personalized movie recommendations to users based on their preferences and movie similarities.

# In[ ]:




