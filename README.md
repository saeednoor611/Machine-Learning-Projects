# Machine-Learning-Projects


Machine learning end to end projects














1 movie recommender system
In this project we will take a movie dataset from kaggle and we will build model that will recommend movies. We will convert this model into website and we will deploy it.

Movie Recommender System
Machine learning-based recommendation systems are powerful engines using machine learning algorithms to segment customers based on their user data and behavioral patterns (such as purchase and browsing history, likes, or reviews) and target them with personalized product and content suggestions.
Here Recommender works as an online seller. It will provide you that product that you are looking for based on you interest. Now a day’s recommender system is used everywhere such as, Facebook YouTube Online Shopping etc

Types of Recommendation system.

1 Content base:
this system recommends contents on the bases of the same content. You want romantic songs then it will provide you more romantic songs. If you are watching love song on YouTube, then YouTube will provide you more love songs.
                                            
2 collaborative filtering 
in this system, contents are being provided on user interest. If user A wants to watch sci-fi movie then user B will must watch sci-fi movie because they have similar interests.
3 hybrid
it is the combination of content base and collaborative filtering base systems.

So in this project we will be using content base recommend system.

Project Flow:
1 Load Data
2 Preprocessing
3 creating model
4 website
5 deploy

Get started!


import pandas as pd
import numpy as np
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

we have two datasets by the name of movies and credits.

movies. Head(3)

budget	genres	homepage	id	keywords	original_language ……………………………….
0	237000000	[{"id": 28, "name": "Action"}, {"id": 12, "nam...	http://www.avatarmovie.com/	19995	[{"id": 1463, "name": "culture clash"}, {"id":.
1	300000000	[{"id": 12, "name": "Adventure"}, {"id": 14, "...	http://disney.go.com/disneypictures/pirates/	285	[{"id": 270, "name": "ocean"}, {"id": 726, "na...
2	245000000	[{"id": 28, "name": "Action"}, {"id": 12, "nam...	http://www.sonypictures.com/movies/spectre/	206647	[{"id": 470, "name": "spy"}, {"id": 818, "name...

In this dataframe we have a lot of columns and will remove some of them and will keep the important one and we will use them for our model.

Now let’s check credit dataframe.
credits. Head(2)
movie_id	title	cast	crew
0	19995	Avatar	[{"cast_id": 242, "character": "Jake Sully", "...	[{"credit_id": "52fe48009251416c750aca23", "de..
1	285	Pirates of the Caribbean: At World's End	[{"cast_id": 4, "character": "Captain Jack Spa...	[{"credit_id": "52fe4232c3a36847f800b579", "de.

If I check the cast count in a move
credits. Head()['cast'].values
array(['[{"cast_id": 242, "character": "Jake Sully", "credit_id": "5602a8a7c3a3685532001c9a", "gender": 2, "id": 65731, "name": "Sam Worthington", "order": 0}, {"cast_id": 3, "character": "Neytiri", "credit_id": "52fe48009251416c750ac9cb", "gender": 1, "id": 8691, ………………………………………………….etc.

# first merge both datasets
movies = movies.merge(credits,on='title')
movies.head(1)

we need to remove some columns
In [16]:
 
# keeping columns list from this dataset
# 1 genres
# 2 movie_id
# 3 keywords
# 4 title
# 5 overview
# 6 cast                 
# 7 crew 
moveis = movies[['id','genres','keywords','title','overview','cast','crew']]
moveis.head(1)

id	genres	keywords	title	overview	cast	crew
0	19995	[{"id": 28, "name": "Action"}, {"id": 12, "nam...	[{"id": 1463, "name": "culture clash"}, {"id":...	Avatar	In the 22nd century, a paraplegic Marine is di...	[{"cast_id": 242, "character": "Jake Sully",.	[{"credit_id": "52fe48009251416c750aca23", "de..


From this data we will create a new data frame with three columns.
Id, title and tags.
Tags can be created from overview, genres, keywords, cast, crew.
But we have to preprocess all these columns to be set in a right format.
Missing Data:
# missing data
moveis.isnull().sum()
Out[18]:
id          0
genres      0
keywords    0
title       0
overview    3
cast        0
crew        0
dtype: int64


moveis.dropna(inplace=True)
moveis.isnull().sum()
id          0
genres      0
keywords    0
title       0
overview    0
cast        0
crew        0
dtype: int64


duplicated data:
#duplicate data
moveis.duplicated().sum()
0

Format the columns:
moveis.iloc[0].genres
Out[23]:
'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'

It is a list of dictionary and it is a weird format. What we want is like this
[‘action’,’adventure’,’fantasy’,sci-fi’]
So we need a helper function with a loop for each dictionary.
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return Lcovert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Input In [26], in <cell line: 1>()
----> 1 count('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

Input In [24], in count(obj)
      2 L = []
      3 for i in obj:
----> 4     L.append(i['name'])
      5 return L

TypeError: string indices must be integers

# this will not work because we have a string so before passing string list we need to convert it into interger list and for 
# we have python module ast      
import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]') 
[{'id': 28, 'name': 'Action'},
 {'id': 12, 'name': 'Adventure'},
 {'id': 14, 'name': 'Fantasy'},
 {'id': 878, 'name': 'Science Fiction'}]
In [ ]:
So what we can do now.
import ast
moveis['genres'] = moveis['genres'].apply(convert)0       [Action, Adventure, Fantasy, Science Fiction]
1                        [Adventure, Fantasy, Action]
2                          [Action, Adventure, Crime]
3                    [Action, Crime, Drama, Thriller]
4                [Action, Adventure, Science Fiction]
                            ...                      
4804                        [Action, Crime, Thriller]
4805                                [Comedy, Romance]
4806               [Comedy, Drama, Romance, TV Movie]
4807                                               []
4808                                    [Documentary]
Name: genres, Length: 4806, dtype: object


Now we will do the same action for keyword column.
# do the same for keywords column
movies['keywords']= movies['keywords'].apply(convert)
0       [culture clash, future, space war, space colon...
1       [ocean, drug abuse, exotic island, east india ...
2       [spy, based on novel, secret agent, sequel, mi...
3       [dc comics, crime fighter, terrorist, secret i...
4       [based on novel, mars, medallion, space travel...
                              ...                        
4804    [united states–mexico barrier, legs, arms, pap...
4805                                                   []
4806    [date, love at first sight, narration, investi...
4807                                                   []
4808            [obsession, camcorder, crush, dream girl]
Name: keywords, Length: 4806, dtype: object


Now for cast column: we need first three actors from first three dictionary.
# now cast column.
moveis['cast'][0]
[{"cast_id": 242, "character": "Jake Sully", "credit_id": "5602a8a7c3a3685532001c9a", "gender": 2, "id": 65731, "name": "Sam Worthington", "order": 0}, {"cast_id": 3, "character": "Neytiri", "credit_id": "52fe48009251416c750ac9cb", "gender": 1, "id": 8691, "name": "Zoe Saldana", "order": 1}, {"cast_id": 25, "character": "Dr. Grace Augustine", "credit_id": "52fe48009251416c750aca39", "gender": 1, "id": 10205, "name": "Sigourney Weaver", "order": 2}, 

For this process we will use the same codes.
# now cast column.
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
             L.append(i['name'])
             counter+=1
            
        else:
            break   
    return Lmoveis['cast'] = moveis['cast'].apply(convert3)
0       [In [ ]:
 


Now crew column; it’s a bit difficult but we will solve it.
# now crew
moveis['crew'][0]
[{"credit_id": "52fe48009251416c750aca23", "department": "Editing", "gender": 0, "id": 1721, "job": "Editor", "name": "Stephen E. Rivkin"}, {"credit_id": "539c47ecc3a36810e3001f87", "department": "Art", "gender": 2, "id": 496, "job": "Production Design", "name": "Rick Carter"}, {"credit_id": "54491c89c3a3680fb4001cf7", "department": "Sound", "gender": 0, "id": 900, "job": "Sound Designer", "name": "Christopher Boyes"}, {"credit_id": "54491cb70e0a267480001bd0", "department": "Sound", "gender": 0, "id": 900, "job": "Supervising Sound Editor", "name": "Christopher Boyes"}, {"credit_id": "539c4a4cc3a36810c9002101", "department": "Production", "gender": 1, "id": 1262, "job": "Casting", "name": "Mali Finn"}, {"credit_id": "5544ee3b925141499f0008fc", "department": "Sound", "gender": 2, "id": 1729, "job": "Original Music Composer", "name": "James Horner"}, {"credit_id": "52fe48009251416c750ac9c3", "department": "Directing", "gender": 2, "id": 2710, "job": "Director", "name": "James Cameron"}, {"credit_id": "52fe48009251416c750ac9d9", "department": "Writing", "gender": 2, "id": 2710, "job": "Writer", "name": "James Cameron"},

We have a lot of dictionaries but we need only director dictionary.
def director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
moveis['crew'].apply(director)
0           [James Cameron]
1          [Gore Verbinski]
2              [Sam Mendes]
3       [Christopher Nolan]
4          [Andrew Stanton]
               ...         
4804     [Robert Rodriguez]
4805         [Edward Burns]
4806          [Scott Smith]
4807          [Daniel Hsia]
4808     [Brian Herzlinger]
Name: crew, Length: 4806, dtype: object



So finally we have formatted all the columns but # one last thing we have to do is, covert overview column into a list
moveis['overview'].apply(lambda x: x.split())
0       [In, the, 22nd, century,, a, paraplegic, Marin...
1       [Captain, Barbossa,, long, believed, to, be, d...
2       [A, cryptic, message, from, Bond’s, past, send...
3       [Following, the, death, of, District, Attorney...
4       [John, Carter, is, a, war-weary,, former, mili...
                              ...                        
4804    [El, Mariachi, just, wants, to, play, his, gui...
4805    [A, newlywed, couple's, honeymoon, is, upended...
4806    ["Signed,, Sealed,, Delivered", introduces, a,...
4807    [When, ambitious, New, York, attorney, Sam, is...
4808    [Ever, since, the, second, grade, when, he, fi...
Name: overview, Length: 4806, dtype: object

In [ ]:

Finally:
But in the last four columns we have to use transformation for removing any space. The problem is all the words are separate from each other we are going to create tags of them. 
So very simple code:
moveis['genres'].apply(lambda x :[i.replace(" ","") for i in x])
0       [Action, Adventure, Fantasy, ScienceFiction]
1                       [Adventure, Fantasy, Action]
2                         [Action, Adventure, Crime]
3                   [Action, Crime, Drama, Thriller]
4                [Action, Adventure, ScienceFiction]
                            ...                     
4804                       [Action, Crime, Thriller]
4805                               [Comedy, Romance]
4806               [Comedy, Drama, Romance, TVMovie]
4807                                              []
4808                                   [Documentary]


moveis['genres'] = moveis['genres'].apply(lambda x :[i.replace(" ","") for i in x])
moveis['keywords'] = moveis['keywords'].apply(lambda x :[i.replace(" ","") for i in x])
moveis['cast'] = moveis['cast'].apply(lambda x :[i.replace(" ","") for i in x])
moveis['crew'] = moveis['crew'].apply(lambda x :[i.replace(" ","") for i in x])
concatenate last five columns
moveis['tags'] = moveis['overview']+moveis['genres']+moveis['keywords']+moveis['cast']+moveis['crew']
now we have to create a new dataframe with 3 columns.

 

	So we have this dataframe.
# in tags we have list so covert it into string
new_df['tags'].apply(lambda x: " ".join(x))
0       In the 22nd century, a paraplegic Marine is di...
1       Captain Barbossa, long believed to be dead, ha...
2       A cryptic message from Bond’s past sends him o...
3       Following the death of District Attorney Harve...
4       John Carter is a war-weary, former military ca...
                              ...                        
4804    El Mariachi just wants to play his guitar and ...
4805    A newlywed couple's honeymoon is upended by th...
4806    "Signed, Sealed, Delivered" introduces a dedic...
4807    When ambitious New York attorney Sam is sent t...
4808    Ever since the second grade when he first saw ...
Name: tags, Length: 4806, dtype: object

new_df['tags'][0]
'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron'


Covert everything into lower case:
new_df['tags'].apply(lambda x:x.lower())
0       in the 22nd century, a paraplegic marine is di...
1       captain barbossa, long believed to be dead, ha...
2       a cryptic message from bond’s past sends him o...
3       following the death of district attorney harve...
4       john carter is a war-weary, former military ca...
                              ...                        
4804    el mariachi just wants to play his guitar and ...
4805    a newlywed couple's honeymoon is upended by th...
4806    "signed, sealed, delivered" introduces a dedic...
4807    when ambitious new york attorney sam is sent t...
4808    ever since the second grade when he first saw ...
Name: tags, Length: 4806, dtype: object







Vectorization:

We are mostly prepare for our model but we need vectorization. For this model we use website. There may user search any particular movie but we have used here content base.

We have to calculate two text similarity.
new_df['tags'][0]
'in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron'
new_df['tags'][1]
"captain barbossa, long believed to be dead, has come back to life and is headed to the edge of the earth with will turner and elizabeth swann. but nothing is quite as it seems. adventure fantasy action ocean drugabuse exoticisland eastindiatradingcompany loveofone'slife traitor shipwreck strongwoman ship alliance calypso afterlife fighter pirate swashbuckler aftercreditsstinger johnnydepp orlandobloom keiraknightley goreverbinski"


So here we have to check the similarity between the text.
Let’s say from list one we have a word “future” 10 times, so we have to check it in the second list it’s count as well.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], 
cv.get_feature_names()
['000',
 '007',
 '10',
 '100',
 '11',
 '12',
 '13',
 '14',
 '15',
 '16',
 '17',
 '18',
 '18th',
 '19',
 '1930s',
 '1940s',
 '1950',
 '1950s',
 '1960s',
 '1970s',
 '1980',
 '1980s',
 '1985',
 '1990s',
 '1999',
 '19th',
 '19thcentury',
 '20',
 '200',
 '2009',
 '20th',
 '24',
 '25',
 '30',
 '300',
 '3d',
 '40',
 '50',
 '500',
 '60',
 '60s',
 '70',
 '70s',
 'aaron',
 'aaroneckhart',
 'abandoned',
 'abducted',
 'abigailbreslin',
 'abilities',
 'ability',
 'able',
 'aboard',
 'abuse',
 'abusive',
 'academy',
 'accept',
 'accepted',
 'accepts',
 'access',
 'accident',
 'accidental',
 'accidentally',
 'accompanied',
 'accomplish',
 'account',
 'accountant',
 'accused',
 'ace',
 'achieve',
 'act',
 'acting',
 'action',
 'actionhero',
 'actions',
 'activist',
 'activities',
 'activity',
 'actor',
 'actors',
 'actress',
 'acts',
 'actual',
 'actually',
 'adam',
 'adams',
 'adamsandler',
 'adamshankman',
 'adaptation',
 'adapted',
 'addict',
 'addicted',
 'addiction',
 'adolescence',
 'adolescent',
 'adopt',
 'adopted',
 'adoption',
 'adopts',
 'adrienbrody',
 'adult',
 'adultanimation',
 'adultery',
 'adulthood',
 'adults',
 'advantage',
 'adventure',
 'adventures',
 'advertising',
 'advice',
 'affair',
 'affairs',
 'affection',
 'affections',
 'afghanistan',
 'africa',
 'african',
 'africanamerican',
 'aftercreditsstinger',
 'afterlife',
 'aftermath',
 'age',
 'aged',
 'agedifference',
 'agency',
 'agenda',
 'agent',
 'agents',
 'aggressive',
 'aging',
 'ago',
 'agree',
 'agrees',
 'ahead',
 'aid',
 'aided',
 'aids',
 'ailing',
 'air',
 'airplane',
 'airplanecrash',
 'airport',
 'aka',
 'al',
 'alabama',
 'alan',
 'alaska',
 'albert',
 'alcohol',
 'alcoholic',
 'alcoholism',
 'alecbaldwin',
 'alex',
 'alfredhitchcock',
 'ali',
 'alice',
 'alien',
 'alieninvasion',
 'alienlife',
 'aliens',
 'alike',
 'alive',
 'allen',
 'alliance',
 'allied',
 'allies',
 'allow',
 'allowing',
 'allows',


In this list we have similar words such as
Love loving or loved
So we have to apply stemming.
# we have similar words so we have to apply stemming for that we need a library
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
# need helper funct
def stm(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)
new_df['tags'].apply(stm)
0       in the 22nd century, a parapleg marin is dispa...
1       captain barbossa, long believ to be dead, ha c...
2       a cryptic messag from bond’ past send him on a...
3       follow the death of district attorney harvey d...
4       john carter is a war-weary, former militari ca...
                              ...                        
4804    el mariachi just want to play hi guitar and ca...
4805    a newlyw couple' honeymoon is upend by the arr...
4806    "signed, sealed, delivered" introduc a dedic q...
4807    when ambiti new york attorney sam is sent to s...
4808    ever sinc the second grade when he first saw h...
Name: tags, Length: 4806, dtype: object
In [ ]:
 



Now we have to calculate distance between vectors
 from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity[0] similarity[0]


now we need a function that will return five similar movies on the basis of a given movie.
# 1st we need sorting
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])
[(0, 1.0000000000000002),
 (1216, 0.28676966733820225),
 (2409, 0.26901379342448517),
 (3730, 0.2605130246476754),
 (507, 0.255608593705383),
 (539, 0.25038669783359574),
 (582, 0.24511108480187255),
 (1204, 0.24455799402225925),
 (1194, 0.2367785320221084),
 (61, 0.23179316248638276),
 (778, 0.23174488732966073),
 (4048, 0.2278389747471728),
 (1920, 0.2252817784447915),
 (2786, 0.21853668936906193),
 (172, 0.21239769762143662),
 (972, 0.2108663315950723),
 (322, 0.2105263157894737),
 (2333, 0.20443988269091456),
Etc.
But what we need that are first five movies.
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
[(1216, 0.28676966733820225),
 (2409, 0.26901379342448517),
 (3730, 0.2605130246476754),
 (507, 0.255608593705383),
 (539, 0.25038669783359574)]
# for this function we need each movie index like,
# new_df[new_df['title']=='Avatar'].index[0]




def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)recommend('Avatar')
Aliens vs Predator: Requiem
Aliens
Falcon Rising
Independence Day
Titan A.E.


For avatar first five movies list.
