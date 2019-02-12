

```python
import nltk
nltk.download('wordnet')
from nltk.corpus import gutenberg, stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as soup, SoupStrainer
import requests
import lxml
from collections import Counter
from requests import exceptions
import os
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from rake_nltk import Rake
import matplotlib.pyplot as plt; plt.rcdefaults()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import steam
from steam.api import interface
steam.api.key.set('FC864146AEF094845C669F86F0B20CFF')
pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_colwidth', 80)
%matplotlib inline
```

    [nltk_data] Downloading package wordnet to /Users/sabol/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    

# Steam Recommendation Engine #
## Overview ##
As someone somewhere probably said, it's the best time to be playing video games.  Democratization of the tools needed and technical know-how has gotten to the point that almost anyone can sit down and within a matter of 80-100 work weeks over 2-4 years (lol) crank out a game.  But with this over-abundance of choice comes a new problem... UNCERTAINTY. "Do I buy that hot new game that just came out?  What about that other new hot game that came out or that older game that everyone said was great?  Hell, maybe I should just play one of the games I already bought but never installed?”  My aim is to build a recommendation engine that will help people (ME) with this.  Specifically, the engine will not just recommend games available on steam, but will also recommend you games that are already in your library but unplayed.  Yay for saving money!

## Data Collection ##
In order to build the recommendation engine using steam, I used BeautifulSoup to crawl the Steam Store for all games with steam pages. I did this in two steps.  Step 1 was generating a list of links to each game's steam page, starting at page 1 (https://store.steampowered.com/search/?category1=998&page=1).  Once I was able to grab every link, I iterated through each one, grabbing relevant fields such as title, image, description, tags, reviews, etc.

### Grabbing each store link ###


```python
# Get title and links for all steam games
urls = ['https://store.steampowered.com/search/?category1=998&page={}'.format(i) for i in range(1, 1185)]

filename = "products_v2.csv"
f = open(filename, "w", encoding="utf-8")
headers = "game_title\tsteam_url\n"
f.write(headers)

for url in urls:
    r = requests.get(url)
    page_soup = soup(r.content, "html.parser")

    containers = page_soup.findAll("a", {"class":"search_result_row ds_collapse_flag "})

    for container in containers:
        game_title = container.findAll("span", {"class":"title"})[0].text
        link = container["href"]

        #print(game_title)
        #print(link)
        #print("----------------")

        f.write(game_title + "\t" + link + "\n")

f.close()
```

### Iterating through each link, grabing all relevant info from the page###


```python
df_links = pd.read_csv('products_v2.csv', delimiter="\t")

tags = []
images = []
descriptions = []
num_reviews = []
rating_values = []
releases = []

cookies = {'birthtime': '568022401'}

for link in df_links.iterrows():
    print(link[1]['game_title'])
    url = link[1]['steam_url']
    r = requests.get(url, cookies=cookies)
    page_soup = soup(r.content, 'lxml')
    try:
        tag = page_soup.find("div", {"class":"glance_tags popular_tags"})
        tag = tag.text
        tag = tag.replace('\t', '').replace('\r', '').replace('\n', ', ').replace('+, ', '').replace(', , ', '')
        tags.append(tag)
    except:
        tag = 'no tags'
        tags.append(tag)
    
    try:
        image = page_soup.find('img', {'class':'game_header_image_full'})['src']
        images.append(image)
    except:
        image = 'no_image'
        images.append(image)
    
    try:
        num_review = int(page_soup.find('meta', {'itemprop':'reviewCount'})['content'])
        num_reviews.append(num_review)
    except:
        num_review = 'no reviews'
        num_reviews.append(num_review)

    try:
        rating_value = float(page_soup.find('meta', {'itemprop':'ratingValue'})['content'])
        rating_values.append(rating_value)
    except:
        rating_value = 'no rating'
        rating_values.append(rating_value)
    
    try:
        description = page_soup.find('div', {'class': 'game_description_snippet'}).text.replace('\t','').replace('\r','').replace('\n', '')
        descriptions.append(description)
    except:
        description = 'no description'
        descriptions.append(description)
        
    try:
        release = page_soup.find('div', {'class': 'date'}).text
        releases.append(release)
    except:
        release = 'no release date'
        releases.append(release)

        
        
    
      
the_rest = pd.DataFrame({'tags': tags, 'image': images, 'description': descriptions,
                         'number_of_review': num_reviews,
                         'score': rating_values,
                         'release_date': releases})

steam_scrape_the_rest = pd.concat([df_links, the_rest], axis=1, sort=False)
steam_scrape_the_rest.to_csv('steam_games_all_fields.csv', sep='\t')
```

## Data Cleaning and Processing ##


```python
steam_games = pd.read_csv('steam_games_all_fields.csv', index_col=0, delimiter='\t')

# Split out 'appid' from the URL string.  This will be used later to merge in my own games
# from the steam api
app_id = steam_games['steam_url'].str.split('/',expand=True)
app_id = app_id.drop([0, 1, 2, 5, 6], axis=1)
app_id.columns = ['app', 'game_ID']
df = pd.concat([steam_games, app_id], axis=1, sort=False)

# Fill nil values and remove random characters
df['description'] = df['description'].fillna('')
df = df[~df.description.str.contains('no description')]
df['release_date'] = df['release_date'].replace({'no release date' : ''})
df['game_title'] = df['game_title'].str.replace(r'®', '')
df['game_title'] = df['game_title'].str.replace(r'™', '')
df['number_of_review'] = df['number_of_review'].replace({'no reviews' : ''})
df['score'] = df['score'].replace({'no rating' : ''})

# Dropping unneeded/dupe columns and renaming others for merge purposes later
df = df.drop(['app'], axis=1)
df['game_ID'] = pd.to_numeric(df['game_ID'], downcast='signed')
df['number_of_review'] = pd.to_numeric(df['number_of_review'], downcast='integer')
df['score'] = pd.to_numeric(df['score'])

# Game must have a score, scores are not calculated until a min number of reviews is satisfied
df = df[df['score'] > 0]
df = df.rename(columns={"game_ID": "appid"})
df = df.rename(columns={"Unnamed: 0": "id"})
df = df.drop_duplicates(subset='appid', keep='first')
df = df.dropna(subset=['game_title'])
df = df.reset_index()
df = df.drop(['index'], axis=1)
print('Dataframe Shape', df.shape)
df.head()
```

    Dataframe Shape (14596, 9)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_title</th>
      <th>steam_url</th>
      <th>tags</th>
      <th>image</th>
      <th>description</th>
      <th>number_of_review</th>
      <th>score</th>
      <th>release_date</th>
      <th>appid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Counter-Strike: Global Offensive</td>
      <td>https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/?snr=1...</td>
      <td>FPS, Multiplayer, Shooter, Action, Team-Based, Competitive, Tactical, First-...</td>
      <td>https://steamcdn-a.akamaihd.net/steam/apps/730/header.jpg?t=1544148568</td>
      <td>Counter-Strike: Global Offensive (CS: GO) expands upon the team-based action...</td>
      <td>2,915,091.00000</td>
      <td>9.00000</td>
      <td>Aug 21, 2012</td>
      <td>730</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MONSTER HUNTER: WORLD</td>
      <td>https://store.steampowered.com/app/582010/MONSTER_HUNTER_WORLD/?snr=1_7_7_23...</td>
      <td>Action, Hunting, Co-op, Open World, Multiplayer, Third Person, RPG, Adventur...</td>
      <td>https://steamcdn-a.akamaihd.net/steam/apps/582010/header.jpg?t=1544082685</td>
      <td>Welcome to a new world! In Monster Hunter: World, the latest installment in ...</td>
      <td>55,314.00000</td>
      <td>6.00000</td>
      <td>Aug 9, 2018</td>
      <td>582010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Path of Exile</td>
      <td>https://store.steampowered.com/app/238960/Path_of_Exile/?snr=1_7_7_230_150_1</td>
      <td>Free to Play, Action RPG, Hack and Slash, RPG, Multiplayer, Massively Multip...</td>
      <td>https://steamcdn-a.akamaihd.net/steam/apps/238960/header.jpg?t=1544390585</td>
      <td>You are an Exile, struggling to survive on the dark continent of Wraeclast, ...</td>
      <td>74,977.00000</td>
      <td>9.00000</td>
      <td>Oct 23, 2013</td>
      <td>238960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Insurgency: Sandstorm</td>
      <td>https://store.steampowered.com/app/581320/Insurgency_Sandstorm/?snr=1_7_7_23...</td>
      <td>FPS, Realistic, Shooter, Multiplayer, Action, Military, Tactical, Singleplay...</td>
      <td>https://steamcdn-a.akamaihd.net/steam/apps/581320/header.jpg?t=1546538316</td>
      <td>Insurgency: Sandstorm is a team-based, tactical FPS based on lethal close qu...</td>
      <td>9,019.00000</td>
      <td>9.00000</td>
      <td>Dec 12, 2018</td>
      <td>581320</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Warframe</td>
      <td>https://store.steampowered.com/app/230410/Warframe/?snr=1_7_7_230_150_1</td>
      <td>Free to Play, Action, Co-op, Multiplayer, Third-Person Shooter, Sci-fi, Ninj...</td>
      <td>https://steamcdn-a.akamaihd.net/steam/apps/230410/header.jpg?t=1545251372</td>
      <td>Warframe is a cooperative free-to-play third person online action game set i...</td>
      <td>236,593.00000</td>
      <td>9.00000</td>
      <td>Mar 25, 2013</td>
      <td>230410</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_desc = df[['number_of_review', 'score']]
df_desc.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_of_review</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14,596.00000</td>
      <td>14,596.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1,709.28008</td>
      <td>7.38113</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28,152.31748</td>
      <td>1.38787</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.00000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.00000</td>
      <td>6.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>76.00000</td>
      <td>7.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>345.00000</td>
      <td>9.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2,915,091.00000</td>
      <td>10.00000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Exploration ##


```python
text = " ".join(desc for desc in df.description)
print ("There are {:,.0f} words in the combination of all descriptions.".format(len(text)))
stopwords = set(STOPWORDS)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
video_mask = np.array(Image.open(path.join(d, "video_game_mask.png")))
wc = WordCloud(stopwords=stopwords, background_color="black", max_words=10000,
                      contour_width=3, mask=video_mask, contour_color='white').generate(text)

wc.to_file(path.join(d, 'video_games.png'))

fig, ax = plt.subplots(figsize=(20, 15))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
```

    There are 3,098,835 words in the combination of all descriptions.
    


![png](Steam%20Recommendation%20Engine_files/Steam%20Recommendation%20Engine_11_1.png)



```python
text = " ".join(desc for desc in df.tags)
textlist = text.split(", ")
wordfreq = [textlist.count(w) for w in textlist]
pairs = Counter(text.split(", ")).most_common(25)
word = []
frequency = []

for i in range(len(pairs)):
    word.append(pairs[i][0])
    frequency.append(pairs[i][1])

indices = np.arange(len(pairs))
fig, ax = plt.subplots(figsize=(15, 10))
plt.bar(indices, frequency)
plt.title('Most Common Steam Tags by Count', )
plt.xticks(indices, word, rotation='vertical')
plt.tight_layout()
plt.show()
```


![png](Steam%20Recommendation%20Engine_files/Steam%20Recommendation%20Engine_12_0.png)



```python
no_out = df[df['number_of_review'] < 2000]
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.boxplot(x="score", y = 'number_of_review', data=no_out, linewidth=2.5)
```


![png](Steam%20Recommendation%20Engine_files/Steam%20Recommendation%20Engine_13_0.png)



```python

sns.jointplot(x='score', y='number_of_review', data=df, height=10, color='g')
#plt.title('Steam Tags by Count')
#plt.xticks(indices, word, rotation='vertical')
plt.show()
```


![png](Steam%20Recommendation%20Engine_files/Steam%20Recommendation%20Engine_14_0.png)


### Using the Steam API to Build My Games DF###
In order to create a recommendation engine that recommends my own games, I needed to find a way to procure my own steam games data.  While I could have also grabbed this data via web scraping, I decided to make use of Steam's API, which allows for easy access into a person's steam games library, provided a user_ID is provided.


```python
my_steam_id = 76561197974553664
games = interface('IPlayerService').GetOwnedGames(steamid=my_steam_id, include_appinfo=1)
my_games = games['response']['games']
game_table = pd.DataFrame(my_games)
game_table = game_table.drop(['has_community_visible_stats', 'playtime_2weeks',
                              'img_icon_url', 'img_logo_url'], axis=1)
# Convert playtime from minutes to hours
game_table['playtime_forever'] = game_table['playtime_forever'] / 60
my_games_info = pd.merge(df, game_table, on='appid')
print('My Games DataFrame Shape:', my_games_info.shape)

my_games_describe = my_games_info[['number_of_review', 'score', 'playtime_forever']]
my_games_describe.describe()
```

    My Games DataFrame Shape: (213, 11)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_of_review</th>
      <th>score</th>
      <th>playtime_forever</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>213.00000</td>
      <td>213.00000</td>
      <td>213.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39,327.68545</td>
      <td>8.64319</td>
      <td>9.38529</td>
    </tr>
    <tr>
      <th>std</th>
      <td>208,823.54301</td>
      <td>1.22644</td>
      <td>20.50645</td>
    </tr>
    <tr>
      <th>min</th>
      <td>180.00000</td>
      <td>5.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3,998.00000</td>
      <td>9.00000</td>
      <td>0.01667</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>11,035.00000</td>
      <td>9.00000</td>
      <td>2.46667</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>23,045.00000</td>
      <td>9.00000</td>
      <td>10.01667</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2,915,091.00000</td>
      <td>10.00000</td>
      <td>188.78333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create DF of all the steam games that I don't own
not_my_games = pd.merge(df, game_table, how='outer', on='appid')
not_my_games = not_my_games[not_my_games['playtime_forever'].isnull()]
not_my_games = not_my_games.drop(['name', 'image', 'playtime_forever'], axis=1)
print('Dataframe Shape:', not_my_games.shape)
```

    Dataframe Shape: (14383, 8)
    


```python
fig, ax = plt.subplots(figsize=(15, 10))
sns.set_context("paper", font_scale=3)
sns.distplot(not_my_games['score'], norm_hist=True, kde=False, label='Not My Games', ax=ax)
sns.distplot(my_games_info['score'], norm_hist=True, kde=False, label='My Games', ax=ax)
plt.title('My Taste is Impeccable')
plt.legend()
plt.show()
```


![png](Steam%20Recommendation%20Engine_files/Steam%20Recommendation%20Engine_18_0.png)



```python
# Removed outliers.  'Counter-Strike' is a free-to-play game with > 2 million reviews
df_csno = my_games_info[~my_games_info.game_title.str.contains('Counter-Strike: Global Offensive')]
df_csno = df_csno[df_csno['playtime_forever'] <= 100]

fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.scatterplot(x="playtime_forever", y="number_of_review", hue="score", 
                     size= 'score', data=df_csno)
plt.title('Game Playtime x Popularity (# of Reviews)', size = 20)
plt.xlabel('Playtime (Hours)', size=15)
plt.ylabel('Number of Reviews for a Game', size=15)
plt.show()
```


![png](Steam%20Recommendation%20Engine_files/Steam%20Recommendation%20Engine_19_0.png)



```python
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.violinplot(x="score", y="playtime_forever", data=my_games_info)
plt.title('Playtime Broken Down By Game Score', size = 20)
plt.xlabel('Score', size=15)
plt.ylabel('Total Playtime', size=15)
plt.show()
```


![png](Steam%20Recommendation%20Engine_files/Steam%20Recommendation%20Engine_20_0.png)


## Building the Recommender ##

Evaluating a recommender is tough.  Since people's choices and preferences are subjective, there aren't a lot of metrics that can tell you whether or not your enginge performed well beyond looking at it and thinking that it did a goog job.  To try and combat this a bit, I built 3 different versions of the engine, one run by steam tags, another by descriptions, and a hybrid of the two. Then within each of those groups I split out the engine to have 1 that selects games you already own and recommends them to you, and another that looks at all the games you don't own.
<br>

For building the actual recommender system, I had originally decided that I would use Steam tags that are associated with every game.  However, because it was easy enough to collect, I also had descriptions for every game.  So I decided to test the recommendation system with three different setups.  The first being description only, the second being tags only, and the third being a hybrid of both.


```python

```

### How the Recommender Works ###
While I will have built a number of different version, all of the recommenders work in roughly the same way.  Text (either game descriptions or steam tags) is taken in and converted into a matrix of token counts via sklearn's CountVectorizer module.  

### Parsing Game Descriptions ###
For building the recommender with text descriptions, I had initially wanted to use sklearn's term frequency inverse document frequency module to break the descriptions down to components and assign specific weights to words, but over the course of testing I found that the countvectorizer module produced much better results across all of the models I built. TFIDF was matching up games with very short descriptions that had I uncommon word in them but nothing else.  Countvectorizer got rid of this problem.


```python
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]
```


```python
cv_desc = df[['appid', 'game_title', 'tags', 'description']]

stop_words = set(STOPWORDS)
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

cv_desc['stemmed'] = cv_desc.description.apply(lambda x: stemmer.stem(x))
cv_desc['tokens'] = cv_desc.stemmed.apply(lambda x: tokenizer.tokenize(x))
cv_desc['lemmed'] = cv_desc.tokens.apply(lemmatize_text)
cv_desc['stopped'] = cv_desc.lemmed.apply(lambda x: [item for item in x if item not in stop_words])
cv_desc['desc']=cv_desc['stopped'].apply(lambda x: ', '.join(map(str, x)))
```


```python
cv_desc['desc'][3]
```




    'insurgency, sandstorm, team, based, tactical, fps, based, lethal, close, quarter, combat, objective, oriented, multiplayer, gameplay, experience, intensity, modern, combat, skill, rewarded, teamwork, win, fight'



After stemming, tokening, lemming, and stopping the description data, I ran it through the Countvectorizer.  I set ngram_range equal to (1, 2) since often games have titles that are multiple words long and I wanted those values to be extracted together.  Once I got the matrix of counts I applyed the cosine_similarity function in order to compute the similarity of each game.


```python
cv = CountVectorizer(analyzer='word', lowercase=True, ngram_range=(1,2))
cv_matrix = cv.fit_transform(cv_desc['desc'])

# Linear kernel provides better results for descriptions
cosine_similarities = cosine_similarity(cv_matrix, cv_matrix)

results_desc = {}

for idx, row in cv_desc.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-500:-1]
    similar_items = [(cosine_similarities[idx][i], cv_desc['appid'][i]) for i in similar_indices]
    results_desc[row['appid']] = similar_items[1:]
    
print('Done!')
```

    Done!
    


```python
def recommend(game, num):
    app_id = cv_desc.loc[cv_desc['game_title'] == game]['appid'].tolist()[0]
    
    print('Recommending ' + str(num) + " games sitting in your Steam library unplayed, that are similar to " + 
          game + '.')

    rez = results_desc[app_id]

    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    my_game_df = pd.merge(sim_score_df, my_games_info, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'name', 'score',
                                  'tags', 'description', 'steam_url', 'image'], axis = 1)
   
    # Filter out games I've played more than an hour of
    my_game_df = my_game_df[my_game_df['playtime_forever'] <= 1]
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("Assassin's Creed Odyssey", 5)
```

    Recommending 5 games sitting in your Steam library unplayed, that are similar to Assassin's Creed Odyssey.
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-12-017db2461e21> in <module>()
         17     return my_game_df.head(num)
         18 
    ---> 19 recommend("Assassin's Creed Odyssey", 5)
    

    <ipython-input-12-017db2461e21> in recommend(game, num)
          5           game + '.')
          6 
    ----> 7     rez = results_desc[app_id]
          8 
          9     sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    

    NameError: name 'results_desc' is not defined



```python
def recommend(game, num):
    app_id = cv_desc.loc[cv_desc['game_title'] == game]['appid'].tolist()[0]
    
    print('Fine, here are ' + str(num) + " games you could buy that are similar to " + 
          game + '.')

    rez = results_desc[app_id]
   
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    
    my_game_df = pd.merge(sim_score_df, not_my_games, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'score',
                                  'tags', 'description'], axis = 1)
    #my_game_df = my_game_df[my_game_df['playtime_forever'] <= 60]
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("Assassin's Creed Odyssey", 5)
```

    Fine, here are 5 games you could buy that are similar to Assassin's Creed Odyssey.
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-616fd471941c> in <module>()
         16     return my_game_df.head(num)
         17 
    ---> 18 recommend("Assassin's Creed Odyssey", 5)
    

    <ipython-input-11-616fd471941c> in recommend(game, num)
          5           game + '.')
          6 
    ----> 7     rez = results_desc[app_id]
          8 
          9     sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    

    NameError: name 'results_desc' is not defined


#### Using Steam Tags to Build the Engine ####
Every game on steam has a set of user-defined tags that it gets associated to.  Here, I used these tags to compute the similarity betwe


```python
count_vec_tags = df[['appid', 'game_title', 'tags', 'description']]
vectorizer = CountVectorizer(analyzer='word', lowercase=True, ngram_range=(1,2))
count_matrix = vectorizer.fit_transform(count_vec_tags['tags'])
cosine_similarities = cosine_similarity(count_matrix, count_matrix)
results_tags = {}

for idx, row in count_vec_tags.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-500:-1]
    similar_items = [(cosine_similarities[idx][i], count_vec_tags['appid'][i]) for i in similar_indices]

    # First item is the item itself, so remove it.
    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
    results_tags[row['appid']] = similar_items[1:]
    
print('Done!')
```

    Done!
    


```python
vectorizer = CountVectorizer(analyzer='word', lowercase=True, ngram_range=(1,2))
count_matrix = vectorizer.fit_transform(count_vec_tags['tags'])
count_matrix.toarray()
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)




```python
count_vec_tags[['game_title','tags']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_title</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Counter-Strike: Global Offensive</td>
      <td>FPS, Multiplayer, Shooter, Action, Team-Based, Competitive, Tactical, First-...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MONSTER HUNTER: WORLD</td>
      <td>Action, Hunting, Co-op, Open World, Multiplayer, Third Person, RPG, Adventur...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Path of Exile</td>
      <td>Free to Play, Action RPG, Hack and Slash, RPG, Multiplayer, Massively Multip...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Insurgency: Sandstorm</td>
      <td>FPS, Realistic, Shooter, Multiplayer, Action, Military, Tactical, Singleplay...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Warframe</td>
      <td>Free to Play, Action, Co-op, Multiplayer, Third-Person Shooter, Sci-fi, Ninj...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Divinity: Original Sin 2 - Definitive Edition</td>
      <td>RPG, Turn-Based, Co-op, Story Rich, Fantasy, Open World, Character Customiza...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Assassin's Creed Odyssey</td>
      <td>Open World, Action, RPG, Singleplayer, Adventure, Assassin, Historical, Stea...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Rocket League</td>
      <td>Multiplayer, Racing, Soccer, Sports, Competitive, Team-Based, Online Co-Op, ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tom Clancy's Rainbow Six Siege</td>
      <td>FPS, Multiplayer, Tactical, Shooter, Action, Team-Based, First-Person, Co-op...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Grand Theft Auto V</td>
      <td>Open World, Action, Multiplayer, Third Person, First-Person, Crime, Shooter,...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Team Fortress 2</td>
      <td>Free to Play, Multiplayer, FPS, Action, Shooter, Class-Based, Team-Based, Fu...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS</td>
      <td>Survival, Shooter, Multiplayer, PvP, Third-Person Shooter, FPS, Action, Batt...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Rust</td>
      <td>Survival, Crafting, Multiplayer, Open World, Sandbox, Building, PvP, Adventu...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Stardew Valley</td>
      <td>RPG, Simulation, Pixel Graphics, Agriculture, Crafting, Relaxing, Sandbox, B...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Warhammer: Vermintide 2</td>
      <td>Co-op, Gore, First-Person, Multiplayer, Action, Violent, Hack and Slash, Dar...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>War Thunder</td>
      <td>Free to Play, World War II, Multiplayer, Simulation, Flight, War, Tanks, Mil...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Total War: WARHAMMER II</td>
      <td>Strategy, Fantasy, RTS, Grand Strategy, Turn-Based Strategy, Multiplayer, Ac...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kingdom Come: Deliverance</td>
      <td>Medieval, Open World, RPG, Realistic, Historical, Singleplayer, First-Person...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ATLAS</td>
      <td>Early Access, Survival, Open World, Pirates, Massively Multiplayer, Early Ac...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Shadow of the Tomb Raider</td>
      <td>Adventure, Action, Lara Croft, Female Protagonist, Third Person, Story Rich,...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>The Elder Scrolls V: Skyrim Special Edition</td>
      <td>Open World, RPG, Adventure, Singleplayer, Fantasy, Character Customization, ...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Black Desert Online</td>
      <td>Massively Multiplayer, MMORPG, Open World, RPG, Character Customization, Fan...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>The Witcher 3: Wild Hunt</td>
      <td>Open World, RPG, Story Rich, Atmospheric, Mature, Fantasy, Adventure, Choice...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Slime Rancher</td>
      <td>Cute, Exploration, Adventure, Colorful, First-Person, Singleplayer, Open Wor...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Dead by Daylight</td>
      <td>Horror, Survival Horror, Multiplayer, Co-op, Survival, Stealth, Gore, Atmosp...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>The Forest</td>
      <td>Survival, Open World, Horror, Crafting, Adventure, Building, First-Person, S...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Beat Saber</td>
      <td>Early Access, VR, Rhythm, Music, Indie, Early Access, Great Soundtrack, Star...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>DRAGON QUEST XI: Echoes of an Elusive Age - Digital Edition of Light</td>
      <td>RPG, JRPG, Turn-Based Combat, Singleplayer, Anime, Fantasy, Turn-Based, Stor...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Tom Clancy's Ghost Recon Wildlands</td>
      <td>Open World, Shooter, Action, Multiplayer, Co-op, Tactical, Stealth, Third-Pe...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Raft</td>
      <td>Early Access, Survival, Crafting, Multiplayer, Co-op, Adventure, Open World,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14566</th>
      <td>Vigil: Blood Bitterness</td>
      <td>Indie, RPG, Puzzle, Singleplayer</td>
    </tr>
    <tr>
      <th>14567</th>
      <td>Pro Rugby Manager 2015</td>
      <td>Sports, Strategy, Simulation</td>
    </tr>
    <tr>
      <th>14568</th>
      <td>Pro Gamer Manager 2</td>
      <td>Early Access, Indie, Simulation, Sports, Strategy, Early Access, Management,...</td>
    </tr>
    <tr>
      <th>14569</th>
      <td>Race To Mars</td>
      <td>Early Access, Strategy, Simulation, Indie, Space, Early Access, Turn-Based, ...</td>
    </tr>
    <tr>
      <th>14570</th>
      <td>Agricultural Simulator: Historical Farming</td>
      <td>Simulation, Agriculture</td>
    </tr>
    <tr>
      <th>14571</th>
      <td>Farming Giant</td>
      <td>Simulation, Agriculture</td>
    </tr>
    <tr>
      <th>14572</th>
      <td>The Tower</td>
      <td>Indie, Adventure, Action, Simulation, Horror, First-Person, Dungeon Crawler</td>
    </tr>
    <tr>
      <th>14573</th>
      <td>Front Page Sports Football</td>
      <td>Sports, Strategy, Football</td>
    </tr>
    <tr>
      <th>14574</th>
      <td>28 Waves Later</td>
      <td>Action, Casual, Zombies</td>
    </tr>
    <tr>
      <th>14575</th>
      <td>World Basketball Tycoon</td>
      <td>Simulation, Management, Sports, Basketball</td>
    </tr>
    <tr>
      <th>14576</th>
      <td>Age of Survival</td>
      <td>Early Access, Survival, Simulation, Early Access, Open World, Multiplayer, A...</td>
    </tr>
    <tr>
      <th>14577</th>
      <td>Game Tycoon 1.5</td>
      <td>Simulation, Strategy, Management, Economy, Casual</td>
    </tr>
    <tr>
      <th>14578</th>
      <td>Airline Tycoon 2</td>
      <td>Simulation, Strategy, Economy, Management</td>
    </tr>
    <tr>
      <th>14579</th>
      <td>World Ship Simulator</td>
      <td>Simulation, Open World, Realistic, Fishing, Singleplayer, Management, Drivin...</td>
    </tr>
    <tr>
      <th>14580</th>
      <td>Citadels</td>
      <td>Strategy, Action, Medieval, RTS</td>
    </tr>
    <tr>
      <th>14581</th>
      <td>The Culling 2</td>
      <td>Memes, Psychological Horror, Sexual Content, Anime, Walking Simulator, Comed...</td>
    </tr>
    <tr>
      <th>14582</th>
      <td>The District</td>
      <td>Survival, Indie, Adventure, Action, Early Access, Open World, Walking Simula...</td>
    </tr>
    <tr>
      <th>14583</th>
      <td>GASP</td>
      <td>Free to Play, Space, Simulation, Action, Survival, Multiplayer, Indie, Strat...</td>
    </tr>
    <tr>
      <th>14584</th>
      <td>Atooms to Moolecules Demo</td>
      <td>Indie, Casual</td>
    </tr>
    <tr>
      <th>14585</th>
      <td>FrostRunner</td>
      <td>Free to Play, Action, Indie, Parkour</td>
    </tr>
    <tr>
      <th>14586</th>
      <td>Terrible Beast from the East</td>
      <td>Action, Massively Multiplayer, RPG, Adventure</td>
    </tr>
    <tr>
      <th>14587</th>
      <td>女巫与六便士 the sibyl and sixpence</td>
      <td>Indie, Casual</td>
    </tr>
    <tr>
      <th>14588</th>
      <td>异霊 异霊 皓月空华</td>
      <td>Free to Play, Adventure, Indie, Casual, RPG, Strategy, Simulation</td>
    </tr>
    <tr>
      <th>14589</th>
      <td>Fur the Game</td>
      <td>Indie, Casual, Adventure</td>
    </tr>
    <tr>
      <th>14590</th>
      <td>Endless Battle</td>
      <td>Free to Play, Action, Massively Multiplayer, Casual, MOBA</td>
    </tr>
    <tr>
      <th>14591</th>
      <td>La Rana</td>
      <td>Free to Play, Indie, Casual, Adventure</td>
    </tr>
    <tr>
      <th>14592</th>
      <td>The Awesome Adventures of Captain Spirit</td>
      <td>Free to Play, Story Rich, Choices Matter, Adventure, Atmospheric, Great Soun...</td>
    </tr>
    <tr>
      <th>14593</th>
      <td>Kunoichi Botan</td>
      <td>Indie, RPG, Sexual Content, Nudity, Anime, JRPG, Memes, Multiple Endings, Fu...</td>
    </tr>
    <tr>
      <th>14594</th>
      <td>Offendron Warrior</td>
      <td>Free to Play, Action, Indie, Retro, Arcade, Shoot 'Em Up, 2D, Shooter, Diffi...</td>
    </tr>
    <tr>
      <th>14595</th>
      <td>Lisa's Memory - 丽莎的记忆</td>
      <td>Early Access, Indie, Early Access, Adventure, RPG</td>
    </tr>
  </tbody>
</table>
<p>14596 rows × 2 columns</p>
</div>




```python
def recommend(game, num):
    app_id = count_vec_tags.loc[count_vec_tags['game_title'] == game]['appid'].tolist()[0]
    
    print('Recommending ' + str(num) + " games sitting in your Steam library unplayed, that are similar to " + 
          game + '.')

    rez = results_tags[app_id]
    #print(rez)
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    #print(df_test)
    my_game_df = pd.merge(sim_score_df, my_games_info, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'name', 'score',
                                  'tags', 'description', 'steam_url', 'image'], axis = 1)
    my_game_df = my_game_df[my_game_df['playtime_forever'] <= 1]
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("DARK SOULS III", 5)
```

    Recommending 5 games sitting in your Steam library unplayed, that are similar to DARK SOULS III.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sim_score</th>
      <th>game_title</th>
      <th>playtime_forever</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.47692</td>
      <td>Fallout 3: Game of the Year Edition</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.44467</td>
      <td>Fallout: New Vegas</td>
      <td>0.66667</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.40469</td>
      <td>Mass Effect 2</td>
      <td>0.58333</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.38742</td>
      <td>Prince of Persia</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.36795</td>
      <td>Darksiders Warmastered Edition</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def recommend(game, num):
    app_id = count_vec_tags.loc[count_vec_tags['game_title'] == game]['appid'].tolist()[0]
    
    print('Fine, here are ' + str(num) + " games you could buy that are similar to " + 
          game + '.')

    rez = results_tags[app_id]
    #print(rez)
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    #print(df_test)
    my_game_df = pd.merge(sim_score_df, not_my_games, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'score',
                                  'tags', 'description'], axis = 1)
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("DARK SOULS III", 10)
```

    Fine, here are 10 games you could buy that are similar to DARK SOULS III.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sim_score</th>
      <th>game_title</th>
      <th>steam_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.71220</td>
      <td>DARK SOULS: REMASTERED</td>
      <td>https://store.steampowered.com/app/570940/DARK_SOULS_REMASTERED/?snr=1_7_7_2...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.61538</td>
      <td>DARK SOULS: Prepare To Die Edition</td>
      <td>https://store.steampowered.com/app/211420/DARK_SOULS_Prepare_To_Die_Edition/...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.58462</td>
      <td>DARK SOULS II: Scholar of the First Sin</td>
      <td>https://store.steampowered.com/app/335300/DARK_SOULS_II_Scholar_of_the_First...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.58133</td>
      <td>Risen</td>
      <td>https://store.steampowered.com/app/40300/Risen/?snr=1_7_7_230_150_95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.55385</td>
      <td>Two Worlds II HD</td>
      <td>https://store.steampowered.com/app/7520/Two_Worlds_II_HD/?snr=1_7_7_230_150_222</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.54903</td>
      <td>Gothic II: Gold Edition</td>
      <td>https://store.steampowered.com/app/39510/Gothic_II_Gold_Edition/?snr=1_7_7_2...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.54854</td>
      <td>Two Worlds Epic Edition</td>
      <td>https://store.steampowered.com/app/1930/Two_Worlds_Epic_Edition/?snr=1_7_7_2...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.54694</td>
      <td>The Elder Scrolls IV: Oblivion Game of the Year Edition</td>
      <td>https://store.steampowered.com/app/22330/The_Elder_Scrolls_IV_Oblivion_Game_...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.54694</td>
      <td>Dragon's Dogma: Dark Arisen</td>
      <td>https://store.steampowered.com/app/367500/Dragons_Dogma_Dark_Arisen/?snr=1_7...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.53158</td>
      <td>Risen 2: Dark Waters</td>
      <td>https://store.steampowered.com/app/40390/Risen_2_Dark_Waters/?snr=1_7_7_230_...</td>
    </tr>
  </tbody>
</table>
</div>



#### Hybrid Descriptions & Steam Tags ####
Now let's mash the two together and see what we get.


```python
cv_hybrid = df[['appid', 'game_title', 'tags', 'description']]

stop_words = set(STOPWORDS)
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

cv_hybrid['stemmed'] = cv_hybrid.description.apply(lambda x: stemmer.stem(x))
cv_hybrid['tokens'] = cv_hybrid.stemmed.apply(lambda x: tokenizer.tokenize(x))
cv_hybrid['lemmed'] = cv_hybrid.tokens.apply(lemmatize_text)
cv_hybrid['stopped'] = cv_hybrid.lemmed.apply(lambda x: [item for item in x if item not in stop_words])
cv_hybrid['desc']=cv_hybrid['stopped'].apply(lambda x: ', '.join(map(str, x)))

```


```python
cv_hybrid['tag_desc'] = cv_hybrid[['tags', 'desc']].apply(lambda x: ' '.join(x), axis=1)
cv = CountVectorizer(analyzer='word', lowercase=True, ngram_range=(1,2))
cv_hybrid_matrix = cv.fit_transform(cv_hybrid['tag_desc'])
cosine_similarities = cosine_similarity(cv_hybrid_matrix, cv_hybrid_matrix)

results_tags_desc = {}

for idx, row in cv_hybrid.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-500:-1]
    similar_items = [(cosine_similarities[idx][i], cv_hybrid['appid'][i]) for i in similar_indices]

    # First item is the item itself, so remove it.
    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
    results_tags_desc[row['appid']] = similar_items[1:]
    
print('done!')

```

    done!
    


```python
def recommend(game, num):
    app_id = cv_hybrid.loc[cv_hybrid['game_title'] == game]['appid'].tolist()[0]
    
    print('Recommending ' + str(num) + " games sitting in your Steam library unplayed, that are similar to " + 
          game + '.')

    rez = results_tags_desc[app_id]
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    my_game_df = pd.merge(sim_score_df, my_games_info, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'name', 'score',
                                  'tags', 'description', 'steam_url', 'image'], axis = 1)
    my_game_df = my_game_df[my_game_df['playtime_forever'] <= 1]
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("Assassin's Creed Odyssey", 5)
```

    Recommending 5 games sitting in your Steam library unplayed, that are similar to Assassin's Creed Odyssey.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sim_score</th>
      <th>game_title</th>
      <th>playtime_forever</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>0.23034</td>
      <td>Mass Effect 2</td>
      <td>0.58333</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.21553</td>
      <td>Fallout: New Vegas</td>
      <td>0.66667</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.21332</td>
      <td>Prince of Persia</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.21096</td>
      <td>Quantum Break</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.20739</td>
      <td>Fallout 3: Game of the Year Edition</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def recommend(game, num):
    app_id = cv_hybrid.loc[cv_hybrid['game_title'] == game]['appid'].tolist()[0]
    
    print('Fine, here are ' + str(num) + " games you could buy that are similar to " + 
          game + '.')

    rez = results_tags_desc[app_id]
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    my_game_df = pd.merge(sim_score_df, not_my_games, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'score',
                                  'tags', 'description'], axis = 1)
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("Assassin's Creed Odyssey", 5)
```

    Fine, here are 5 games you could buy that are similar to Assassin's Creed Odyssey.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sim_score</th>
      <th>game_title</th>
      <th>steam_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.34221</td>
      <td>Assassin's Creed Origins</td>
      <td>https://store.steampowered.com/app/582160/Assassins_Creed_Origins/?snr=1_7_7...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.32938</td>
      <td>Assassin’s Creed Brotherhood</td>
      <td>https://store.steampowered.com/app/48190/Assassins_Creed_Brotherhood/?snr=1_...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.31906</td>
      <td>Assassin's Creed Revelations</td>
      <td>https://store.steampowered.com/app/201870/Assassins_Creed_Revelations/?snr=1...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.31555</td>
      <td>Assassin's Creed Freedom Cry</td>
      <td>https://store.steampowered.com/app/277590/Assassins_Creed_Freedom_Cry/?snr=1...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31500</td>
      <td>Assassin's Creed Syndicate</td>
      <td>https://store.steampowered.com/app/368500/Assassins_Creed_Syndicate/?snr=1_7...</td>
    </tr>
  </tbody>
</table>
</div>



Hybrid certainly looks good.  Let's test it out on some others just to see.


```python
def recommend(game, num):
    app_id = cv_hybrid.loc[cv_hybrid['game_title'] == game]['appid'].tolist()[0]
    
    print('Recommending ' + str(num) + " games sitting in your Steam library unplayed, that are similar to " + 
          game + '.')

    rez = results_tags_desc[app_id]
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    my_game_df = pd.merge(sim_score_df, my_games_info, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'name', 'score',
                                  'tags', 'description', 'steam_url', 'image'], axis = 1)
    my_game_df = my_game_df[my_game_df['playtime_forever'] <= 1]
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("Rocket League", 10)
```

    Recommending 10 games sitting in your Steam library unplayed, that are similar to Rocket League.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sim_score</th>
      <th>game_title</th>
      <th>playtime_forever</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.43552</td>
      <td>Trine 2: Complete Story</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.41902</td>
      <td>Monaco: What's Yours Is Mine</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.41270</td>
      <td>Trine Enchanted Edition</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.38521</td>
      <td>Lethal League</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.37952</td>
      <td>Awesomenauts - the 2D moba</td>
      <td>0.01667</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.36714</td>
      <td>Mount Your Friends</td>
      <td>0.23333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.36706</td>
      <td>Castle Crashers</td>
      <td>0.28333</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.31260</td>
      <td>Lara Croft and the Guardian of Light</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.27832</td>
      <td>Guacamelee! Gold Edition</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.25652</td>
      <td>Octodad: Dadliest Catch</td>
      <td>0.83333</td>
    </tr>
  </tbody>
</table>
</div>




```python
def recommend(game, num):
    app_id = cv_hybrid.loc[cv_hybrid['game_title'] == game]['appid'].tolist()[0]
    
    print('Fine, here are ' + str(num) + " games you could buy that are similar to " + 
          game + '.')

    rez = results_tags_desc[app_id]
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    my_game_df = pd.merge(sim_score_df, not_my_games, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'score',
                                  'tags', 'description'], axis = 1)
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("Rocket League", 10)
```

    Fine, here are 10 games you could buy that are similar to Rocket League.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sim_score</th>
      <th>game_title</th>
      <th>steam_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.47204</td>
      <td>Move or Die</td>
      <td>https://store.steampowered.com/app/323850/Move_or_Die/?snr=1_7_7_230_150_16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.47031</td>
      <td>Stick Fight: The Game</td>
      <td>https://store.steampowered.com/app/674940/Stick_Fight_The_Game/?snr=1_7_7_23...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.46111</td>
      <td>Bloody Trapland</td>
      <td>https://store.steampowered.com/app/257750/Bloody_Trapland/?snr=1_7_7_230_150_74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.45294</td>
      <td>Sonic &amp; All-Stars Racing Transformed Collection</td>
      <td>https://store.steampowered.com/app/212480/Sonic__AllStars_Racing_Transformed...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.44901</td>
      <td>Robot Roller-Derby Disco Dodgeball</td>
      <td>https://store.steampowered.com/app/270450/Robot_RollerDerby_Disco_Dodgeball/...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.44169</td>
      <td>Hacktag</td>
      <td>https://store.steampowered.com/app/622770/Hacktag/?snr=1_7_7_230_150_181</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.44113</td>
      <td>Overcooked</td>
      <td>https://store.steampowered.com/app/448510/Overcooked/?snr=1_7_7_230_150_14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.43618</td>
      <td>Overcooked! 2</td>
      <td>https://store.steampowered.com/app/728880/Overcooked_2/?snr=1_7_7_230_150_2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.42864</td>
      <td>ROCKETSROCKETSROCKETS</td>
      <td>https://store.steampowered.com/app/289760/ROCKETSROCKETSROCKETS/?snr=1_7_7_2...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.42447</td>
      <td>BattleBlock Theater</td>
      <td>https://store.steampowered.com/app/238460/BattleBlock_Theater/?snr=1_7_7_230...</td>
    </tr>
  </tbody>
</table>
</div>




```python
def recommend(game, num):
    app_id = cv_hybrid.loc[cv_hybrid['game_title'] == game]['appid'].tolist()[0]
    
    print('Recommending ' + str(num) + " games sitting in your Steam library unplayed, that are similar to " + 
          game + '.')

    rez = results_tags_desc[app_id]
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    my_game_df = pd.merge(sim_score_df, my_games_info, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'name', 'score',
                                  'tags', 'description', 'steam_url', 'image'], axis = 1)
    my_game_df = my_game_df[my_game_df['playtime_forever'] <= 1]
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("PLAYERUNKNOWN'S BATTLEGROUNDS", 10)
```

    Recommending 10 games sitting in your Steam library unplayed, that are similar to PLAYERUNKNOWN'S BATTLEGROUNDS.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sim_score</th>
      <th>game_title</th>
      <th>playtime_forever</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.46357</td>
      <td>H1Z1</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.36225</td>
      <td>Lead and Gold: Gangs of the Wild West</td>
      <td>0.68333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.35001</td>
      <td>Star Wars: Battlefront 2 (Classic, 2005)</td>
      <td>0.13333</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.33959</td>
      <td>Warhammer 40,000: Space Marine</td>
      <td>0.30000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.32108</td>
      <td>Red Faction: Armageddon</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.28841</td>
      <td>Red Orchestra 2: Heroes of Stalingrad with Rising Storm</td>
      <td>0.20000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.28370</td>
      <td>Day of Defeat: Source</td>
      <td>0.15000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.27832</td>
      <td>Lara Croft and the Guardian of Light</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.24291</td>
      <td>Alan Wake's American Nightmare</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.24249</td>
      <td>Trine 2: Complete Story</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def recommend(game, num):
    app_id = cv_hybrid.loc[cv_hybrid['game_title'] == game]['appid'].tolist()[0]
    
    print('Fine, here are ' + str(num) + " games you could buy that are similar to " + 
          game + '.')

    rez = results_tags_desc[app_id]
    sim_score_df = pd.DataFrame(rez, columns=['sim_score', 'appid'])
    my_game_df = pd.merge(sim_score_df, not_my_games, how='inner', on='appid')
    my_game_df = my_game_df.drop(['appid', 'number_of_review', 'release_date', 'score',
                                  'tags', 'description'], axis = 1)
    my_game_df = my_game_df.sort_values(['sim_score'], ascending=False)
    return my_game_df.head(num)

recommend("PLAYERUNKNOWN'S BATTLEGROUNDS", 10)
```

    Fine, here are 10 games you could buy that are similar to PLAYERUNKNOWN'S BATTLEGROUNDS.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sim_score</th>
      <th>game_title</th>
      <th>steam_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.54247</td>
      <td>Pixel Royale</td>
      <td>https://store.steampowered.com/app/931250/Pixel_Royale/?snr=1_7_7_230_150_934</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.54149</td>
      <td>Realm Royale</td>
      <td>https://store.steampowered.com/app/813820/Realm_Royale/?snr=1_7_7_230_150_31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.50219</td>
      <td>Ring of Elysium</td>
      <td>https://store.steampowered.com/app/755790/Ring_of_Elysium/?snr=1_7_7_230_150_9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.49696</td>
      <td>Freefall Tournament</td>
      <td>https://store.steampowered.com/app/849940/Freefall_Tournament/?snr=1_7_7_230...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.47289</td>
      <td>Infected Battlegrounds</td>
      <td>https://store.steampowered.com/app/843730/Infected_Battlegrounds/?snr=1_7_7_...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.46973</td>
      <td>ORION: Prelude</td>
      <td>https://store.steampowered.com/app/104900/ORION_Prelude/?snr=1_7_7_230_150_204</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.46847</td>
      <td>Primal Carnage: Extinction</td>
      <td>https://store.steampowered.com/app/321360/Primal_Carnage_Extinction/?snr=1_7...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.46064</td>
      <td>Holdfast: Nations At War</td>
      <td>https://store.steampowered.com/app/589290/Holdfast_Nations_At_War/?snr=1_7_7...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.45648</td>
      <td>Until None Remain: Battle Royale PC Edition</td>
      <td>https://store.steampowered.com/app/697010/Until_None_Remain_Battle_Royale_PC...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.44967</td>
      <td>Until None Remain: Battle Royale VR</td>
      <td>https://store.steampowered.com/app/697020/Until_None_Remain_Battle_Royale_VR...</td>
    </tr>
  </tbody>
</table>
</div>



### I Declare Hybrid The Winner ###
While all three of the models performed well and gave me results that I would consider well within the normal range of what I might consider to be a good recommendation, I think the hybrid system does slightly better than the tag-only system, so it gets the edge.  The fact that it was able to see that I fed it an Assassin's Creed game and recommend me literally EVERY single one that I don't own (on steam) is commendable.  It also performed well when I fed it a party/competative game (Rocket League) and a large scale shooter (PlayerUnknown's Battlegrounds). 

<br>
The tag-only system also did well and, one could, say that it is actually a better performer cause it give some variety, but I don't want variety I want games that I'm going to like!  Description-only comes in last, not surprising considering that most game descriptions on the steam store can be fairly generic and short.

### Evaluation ###
While evalulation a content-based filtering system is tough due to the subjective nature of recommendations, we can do a little bit to prove that the recommender is basic than something more basic.
<br>

Below I wrote out two different simple models that output a list of games at random, for both the games in my steam list and also any and all games in the steam store.  In the second group, I sorted the order by a metric I made up called "popularity", which is simply the score a game has multiplied by it's number of reviews.  Let's see how they did.


```python
rando_games = df.iloc[np.random.choice(np.arange(len(df)), 10, False)]
print('Random List of Steam Games', '\n')
print(rando_games[['game_title', 'score']].head())
print('------------------------------------------------------------------------------')
my_games_rando = my_games_info[my_games_info['playtime_forever'] <= 1]
my_games_rando = my_games_rando.iloc[np.random.choice(np.arange(len(my_games_rando)), 10, False)]
print('Random List of My Games:')
print(my_games_rando[['game_title', 'score']].head())
```

    Random List of Steam Games 
    
                                   game_title   score
    2661                   Forward to the Sky 9.00000
    3237                            Paperbark 8.00000
    4546   Precipice of Darkness, Episode Two 9.00000
    10149       Truck Mechanic Simulator 2015 6.00000
    2341     Serious Sam Classics: Revolution 9.00000
    ------------------------------------------------------------------------------
    Random List of My Games:
                                                      game_title    score
    102                                            Quantum Break  9.00000
    154  Red Orchestra 2: Heroes of Stalingrad with Rising Storm  9.00000
    122                                              Antichamber 10.00000
    196                                       TrackMania² Canyon  7.00000
    76                                               Half-Life 2 10.00000
    


```python
df['popularity'] = df['score'] * df['number_of_review']
rando_games = df.iloc[np.random.choice(np.arange(len(df)), 10, False)]
rando_games = rando_games.sort_values(['popularity'], ascending=False)
print('Random List of Steam Games Sorted By Popularity:')
print(rando_games[['game_title', 'popularity']].head(), '\n')
print('------------------------------------------------------------------------------')
my_games_info['popularity'] = my_games_info['score'] * my_games_info['number_of_review']
my_games_rando = my_games_info[my_games_info['playtime_forever'] <= 1]
my_games_rando = my_games_rando.iloc[np.random.choice(np.arange(len(my_games_rando)), 10, False)]
my_games_rando = my_games_rando.sort_values(['popularity'], ascending=False)
print('Random List of Games I Own Sorted By Popularity:')
print(my_games_rando[['game_title', 'popularity']].head())
```

    Random List of Steam Games Sorted By Popularity:
                             game_title    popularity
    254                Prison Architect 245,655.00000
    471   Grand Theft Auto: San Andreas 215,163.00000
    7088         SCP: Secret Laboratory 167,643.00000
    5721         Alien Rage - Unlimited   5,131.00000
    4560               Worms Crazy Golf   1,863.00000 
    
    ------------------------------------------------------------------------------
    Random List of Games I Own Sorted By Popularity:
                      game_title   popularity
    102            Quantum Break 74,322.00000
    200  Half-Life 2: Lost Coast 41,679.00000
    143                    Braid 38,502.00000
    165              The Swapper 36,340.00000
    212              Dear Esther 26,964.00000
    

As expected, the completely random one doesn't fair too well.  Since there are > 14k steam games, there's is a lot of variabiilty in the types of games you're going to get each time you run it.  Also, since it's strictly random, it does not take into account any preference you might have.  The ordered by popularity random list fairs slightly better in that, at least you're getting fed games of a, generally, high quality.  However, as before, this model does not care about your preferences, if you are looking to play a shooting game, this model is just as likely to recommend you a narrative text adventure as it would a shooter (or anything else).

## Running in Production + Conclusions ##
To run this fully in production I would make a couple changes.  For starters I would not include the scraping utility in the callable program since it takes longer to run than any reasonable person would want to wait.  Ideally that would be something that gets run once a week, with an updated version of the dataset replacing the old, so that whenver the program is run it's fetching the newest data.  I would also not have 3 different versions of this program, only the hybrid would remain since it produced the best results.  For a user to run the program, a web-based UI would probably work best. It would need two inputs in order to run, the steam_id of the person, as well as a game they want to use to seed the recommender.  A v2 implementation would have an option to auto-seed based on the user's most played game, but we're still on v1 right now...
<br>

Overall I would say that I'm fairly pleased with the results I was able to get.  My original vision for this project was to be able to create a program that would recommend me a game that I already owned but hadn't played, which I have succeeded in doing.  Areas of improvement though would be around dataset size as well as automatically inputting whatever your most played game is (or group of games).  While Steam is the largest gaming platform on PC, there are many other platforms with big games that were left out of this dataset. Either scraping those sites and incorporating them into the overall dataset, or finding and utilizing a 3rd party database API would result in a bigger set of data and better recommendations.


```python

```
