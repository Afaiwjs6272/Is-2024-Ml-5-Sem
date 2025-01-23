from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup as BS
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# Setup Chrome WebDriver
drive = 'C:\\Program Files\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe'
options = Options()
service = Service(drive)
driver = webdriver.Chrome(service=service, options=options)

# Open the IMDb page
driver.get('https://www.imdb.com/list/ls538693646/')

# Scroll to load all content
data = []
scroll_pause_time = 1
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Parse the page source
soup = BS(driver.page_source, 'html.parser')
movies = soup.find_all('div', class_="sc-54004b59-3 eOhSkH dli-parent")

for movie in movies:
    title = movie.find('h3', class_='ipc-title__text')
    title_value = title.text.strip() if title else "None"

    rating_star = movie.find("span", class_="ipc-rating-star--rating")
    rating_value = float(rating_star.text.strip().replace(',', '.')) if rating_star else 0.0

    spans = movie.find_all('span', class_='sc-b189961a-8 hCbzGp dli-title-metadata-item')
    year_value = None
    if spans:
        year_text = spans[0].text.strip()
        if '-' in year_text:
            start_year, end_year = map(int, year_text.split('-'))
            year_value = (start_year + end_year) / 2
        else:
            try:
                year_value = int(year_text)
            except ValueError:
                pass

    duration_value = spans[1].text.strip().split()[0] if len(spans) > 1 else "N/A"

    votes_element = movie.find("span", class_="ipc-rating-star--voteCount")
    votes_text = votes_element.text.strip() if votes_element else "0"

    votes_pattern = re.compile(r'(\d+)(K|M)')
    votes_match = votes_pattern.search(votes_text)
    if votes_match:
        votes_number = int(votes_match.group(1))
        votes_suffix = votes_match.group(2).lower()
        if votes_suffix == 'k':
            votes_number *= 1000
        elif votes_suffix == 'm':
            votes_number *= 1000000
    else:
        votes_number = int(votes_text.replace(',', ''))

    director = movie.find("span", class_="sc-74bf520e-5 eesgaX")
    director_value = director.text.strip() if director else "Unknown"

    metaScore = movie.find("span", class_="sc-b0901df4-0 bXIOoL metacritic-score-box")
    metaScore_value = float(metaScore.text.strip().replace(',', '.')) if metaScore else 0.0

    description = movie.find("div", class_="ipc-html-content-inner-div").text

    # Categorize the rating as text
    if rating_value >= 8.0:
        rating_category = "Excellent"
    elif rating_value >= 6.5:
        rating_category = "Good"
    elif rating_value >= 5.0:
        rating_category = "Average"
    else:
        rating_category = "Poor"

    data.append([title_value, rating_value, year_value, duration_value, votes_number, director_value, metaScore_value, description, rating_category])

# Close the WebDriver
driver.quit()

# Create DataFrame
df = pd.DataFrame(data, columns=['Title', 'Rating', 'Year', 'Duration', 'Votes', 'Director', 'MetaScore', 'Description', 'Rating Category'])

# Preprocess the text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = tfidf_vectorizer.fit_transform(df['Director'] + " " + df['Description'])

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Rating Category'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = make_pipeline(MultinomialNB())
model.fit(X_train, y_train)

# Predict the categories
df['Predicted Rating Category'] = label_encoder.inverse_transform(model.predict(X_text))

# Save the DataFrame to CSV
df.to_csv('imdb_movie_data.csv', index=False)

print(f"Данные сохранены в файл imdb_movie_data.csv")