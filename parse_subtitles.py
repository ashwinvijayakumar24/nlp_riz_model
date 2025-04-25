import re
import pandas as pd

def parse_srt(file_path, movie_title, year):
    with open(f"subtitles/{file_path}", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    dialogue = []
    for line in lines:
        line = line.strip()
        if ("-->" not in line and line != "") and (line.isdigit() == False):
            dialogue.append(line)

    df = pd.DataFrame()
    df["Movie Title"] = [movie_title] * len(dialogue) 
    df["Year"] = [year] * len(dialogue) 
    df["Dialogue"] = dialogue
    
    print(df.head())  # Display first few rows
    return df


movie_dict = {
    2004: ['Veer Zaara', 'Main Hoon Na', "Dhoom"], 
    2005: ['Bunty Aur Babli', 'Mangal Pandey', 'Sarkar'],
    2006: ['Dhoom 2', 'Don', 'Fanaa'],
    2007: ['Om Shanti Om', 'Chak De India', 'Taare Zameen Par'],
    2008:['Jodhaa Ahkbar', 'Sarkar Raj', 'Thoda Pyaar Thoda Magic'],
    2009: ['3 Idiots', 'Ajab Prem Ki Ghazab Kahani', 'Wanted' ],
    2010: ['My Name is Khan', 'Tees Maar Khan', 'Once Upon a Time in Mumbaai' ],
    2011: ['Don 2', 'Zindagi Na Milegi Dobara', 'Rockstar'],
    2012: ['Ek Tha Tiger', 'Jab Tak Hai Jaan', 'Bol Bachchan'],
    2013: ['Dhoom 3', 'Race 2', 'Bhaag Milka Bhaag'],
    2014: ['PK', 'Bang Bang', 'Singham Returns'],
    2015: ['Bajrangi Bhaijaan', 'Dilwale', 'Bajirao Mastani'],
    2016: ['Sultan', 'Ae Dil Hai Mushkil', 'Airlift'],
    2017: ['Secret Superstar', 'Tiger Zinda Hai', 'Raees'],
    2018: ['Sanju', 'Padmaavat', 'Thugs of Hindostan'],
    2019: ['War', 'Uri - The Surgical Strike', 'Bharat'],
    2020: ['Tanhaji - The Unsung Warrior', 'Baaghi 3', 'Street Dancer 3D'],
    2021: ['Sooryavanshi', 'Bell Bottom', 'Mumbai Saga'],
    2022: ['The Kashmir Files', 'Gangubai Kathiawadi', 'Laal Singh Chaddha'],
    2023: ['Jawan', 'Pathaan', 'Animal'],
}


all_movies_df = pd.DataFrame()

# Loop through dictionary and process each movie
for year, movies in movie_dict.items():
    for movie in movies:
        file_name = movie.replace(" ", "_") + ".srt"  # Convert to file format
        try:
            df = parse_srt(file_name, movie, year)  # Parse subtitles
            all_movies_df = pd.concat([all_movies_df, df])  # Append to main DataFrame
        except FileNotFoundError:
            print(f"Warning: Subtitle file '{file_name}' not found for {movie} ({year})")

# Save to CSV
all_movies_df.to_csv("structured_dialogues.csv", index=False)

print("CSV file updated successfully!")


