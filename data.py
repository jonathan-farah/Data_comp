import nltk
from nltk.chat.util import Chat, reflections
import requests
from bs4 import BeautifulSoup
import pandas as pd
import tqdm.notebook as tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.sentiment import SentimentIntensityAnalyzer


def scrape_data_from_website(url):
    # Function to scrape data from a given website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Modify this part to extract the specific data you need from the website
    # For example, let's extract the title of the webpage
    title = soup.title.string if soup.title else 'No title found'

    return str(title)

def text_class(title):
     res = {}
     sia = SentimentIntensityAnalyzer
     for index, row in tqdm(title):
        sia.polarity_scores("title")


def main():
    # Load Excel file
    excel_file_path = 'C:\\Users\\jonat\\Downloads\\Dataset_10k.csv'
    df = pd.read_csv('C:\\Users\\jonat\\Downloads\\Dataset_10k.csv')
    df = df.head(10)
    arr =[]
    # Assuming the URLs are in column A starting from row 2
    for index, row in df.iterrows():
        url = row["link"]
        try:
            # Scrape data from the website
            scraped_data = scrape_data_from_website(url)
            
            # Assuming you want to store the scraped data in column B
            arr.append([url, scraped_data])

            print(f"Scraped data from {url}: {scraped_data}")
            print(text_class(scraped_data))

        except Exception as e:
            print(f"Error scraping data from {url}: {e}")
    output = pd.DataFrame(arr,columns=["url","data"])
    print(output.head())

if __name__ == "__main__":
    main()
