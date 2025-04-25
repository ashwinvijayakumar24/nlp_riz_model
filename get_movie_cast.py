import imdb
import pandas as pd

ia = imdb.Cinemagoer()

movie_dict = {
    2004: ['Veer Zaara', 'Main Hoon Na', "Dhoom"], 
    2005: ['Bunty Aur Babli', 'Mangal Pandey', 'Sarkar'],
    2006: ['Dhoom 2', 'Don', 'Fanaa'],
    2007: ['Om Shanti Om', 'Chak De India', 'Taare Zameen Par'],
    2008: ['Jodhaa Ahkbar', 'Sarkar Raj', 'Thoda Pyaar Thoda Magic'],
    2009: ['3 Idiots', 'Ajab Prem Ki Ghazab Kahani', 'Wanted'],
    2010: ['My Name is Khan', 'Tees Maar Khan', 'Once Upon a Time in Mumbaai'],
    2011: ['Don 2', 'Zindagi Na Milegi Dobara', 'Rockstar'],
    2012: ['Ek Tha Tiger', 'Jab Tak Hai Jaan', 'Bol Bachchan'],
}

movie_cast_data = []

for year, movies in movie_dict.items(): 
    for movie_title in movies:
        try:
            search_results = ia.search_movie(movie_title)
            if not search_results:
                print(f"Movie not found: {movie_title} ({year})")
                continue
            if movie_title == "Dhoom" or movie_title == "Sarkar" or movie_title == "Don":
                movie_id = search_results[1].movieID
            elif movie_title == "Wanted":
                movie_id = search_results[4].movieID
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