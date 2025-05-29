import imdb
import pandas as pd

ia = imdb.Cinemagoer()

# https://www.boxofficeindia.com/worldwide-total-gross.php?year=2004
movie_dict = {
    # the top 5 highest-grossing worldwide Bollywood movies as per boxofficeindia.com will not include movies released in languages other than hindi
    # 2004: ['Veer Zaara', 'Main Hoon Na', "Mujhse Shaadi Karogi", "Dhoom", "Khakee"], 
    # 2005: ['No Entry', 'Bunty Aur Babli', 'Saalam Namaste', 'Garam Masala', 'Mangal Pandey -  The Rising'],
    # 2006: ['Dhoom 2', 'Krrish', 'Lage Raho Munnabhai', 'Kabhi Alvida Naa Kehna', 'Don - The Chase Begins Again'],
    # 2007: ['Om Shanti Om', 'Welcome', 'Chak de India', 'Partner', 'Taare Zameen Par'],
    # 2008: ['Ghajini', 'Rab Ne Bana Di Jodi', 'Singh is Kinng', 'Jodhaa Akbar', 'Race'],
    # 2009: ['3 Idiots', 'Love Aaj Kal', 'Ajab Prem Ki Ghazab Kahani', 'Wanted', 'Kambakkht Ishq'],
    # 2010: ['Dabangg', 'My Name is Khan', 'Golmaal 3', 'Raajneeti', 'Housefull'],
    # 2011: ['Bodyguard', 'Ra.One', 'Don 2', 'Ready', 'Zindagi Na Milegi Dobara'],
    # 2012: ['Ek Tha Tiger', 'Dabangg 2', 'Jab Tak Hai Jaan', 'Rowdy Rathore', 'Agneepath'],
    # 2013: ['Dhoom 3', 'Chennai Express', 'Yeh Jawaani Hai Deewani', 'Krrish 3', 'Goliyon Ki Raasleela Ram - Leela'],
    # 2014: ['PK','Kick', 'Happy New Year', 'Bang Bang', 'Singham Returns'],
    # 2015: ['Bajrangi Bhaijaan', 'Dilwale', 'Prem Ratan Dhan Payo', 'Bajirao Mastani', 'Tanu Weds Manu Returns'],
    # 2016: ['Dangal', 'Sultan', 'Ae Dil Hai Mushkil', 'Rustom', 'Airlift'],
    # 2017: ['Tiger Zinda Hai', 'Golmaal Again', 'Raees', 'Tubelight', 'Judwaa 2'],
    # 2018: ['Sanju', 'Padmaavat', 'Simmba', 'Race 3', 'Baaghi 2'],
    # 2019: ['War', 'Kabir Singh', 'Uri - The Surgical Strike', 'Bharat', 'Good Newwz'],
    # 2020: ['Tanhaji - The Unsung Warrior', 'Baaghi 3', 'Street Dancer 3D', 'Malang', 'Shubh Mangal Zyada Saavdhan'],
    # 2021: ['Sooryavanshi', '83', 'Antim: The Final Truth', 'Bell Bottom', 'Tadap'],
    # 2022: ['Brahmastra - Part One - Shiva', 'Bhool Bhulaiyaa 2', 'The Kashmir Files', 'Gangubai Kathiawadi', 'Laal Singh Chaddha'],
    # 2023: ['Jawan', 'Animal', 'Pathaan', 'Gadar 2', 'Tiger 3']
}

movie_cast_data = []

for year, movies in movie_dict.items(): 
    for movie_title in movies:
        try:
            search_results = ia.search_movie(movie_title)
            if not search_results:
                print(f"Movie not found: {movie_title} ({year})")
                continue
            if movie_title == "Dhoom" or movie_title == "Kick":
                movie_id = search_results[1].movieID
            elif movie_title == "Race" or movie_title == "Bodyguard" or movie_title == "Ready" or movie_title == "Bang Bang":
                movie_id = search_results[3].movieID
            elif movie_title == "Wanted":
                movie_id = search_results[4].movieID
            elif movie_title == "Partner":
                movie_id = search_results[5].movieID
            elif movie_title == "Welcome":
                movie_id = search_results[6].movieID
            elif movie_title == "War":
                movie_id = search_results[9].movieID           
            else:
                movie_id = search_results[0].movieID

            movie = ia.get_movie(movie_id)

            for person in movie.get('cast', [])[:10]: 
                actor_name = person.get('name')
                character_name = str(person.currentRole)
                movie_cast_data.append({
                    "Movie Title": movie_title,
                    "Year": year,
                    "Actor": actor_name,
                    "Character": character_name
                })

        except Exception as e:
            print(f"Error fetching {movie_title} ({year}): {e}")

df_cast = pd.DataFrame(movie_cast_data)
df_cast.to_csv("movie_characters.csv", index=False)

print("Cast members successfully saved")
