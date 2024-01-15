<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
</p>
<p align="center">
    <h1 align="center">.</h1>
</p>
<p align="center">
    <em>TechGuru: Simplifying Software Engineering with Power and Precision</em>
</p>
<p align="center">
	<!-- local repository, no metadata badges. -->
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=default&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=default&logo=tqdm&logoColor=black" alt="tqdm">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=default&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=default&logo=YAML&logoColor=white" alt="YAML">
	<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=default&logo=OpenAI&logoColor=white" alt="OpenAI">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<br>
	<img src="https://img.shields.io/badge/AIOHTTP-2C5BB4.svg?style=default&logo=AIOHTTP&logoColor=white" alt="AIOHTTP">
	<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=default&logo=Docker&logoColor=white" alt="Docker">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=default&logo=FastAPI&logoColor=white" alt="FastAPI">
	<img src="https://img.shields.io/badge/JSON-000000.svg?style=default&logo=JSON&logoColor=white" alt="JSON">
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Features](#-features)
> - [ Repository Structure](#-repository-structure)
> - [ Modules](#-modules)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [ Running .](#-running-.)
>   - [ Tests](#-tests)
> - [ Project Roadmap](#-project-roadmap)
> - [ Contributing](#-contributing)
> - [ License](#-license)
> - [ Acknowledgments](#-acknowledgments)

---

##  Overview

The codebase is a project aimed at fetching and analyzing data from MongoDB. It utilizes various libraries such as argparse, asyncio, pandas, pymongo, and telethon for efficient data handling and manipulation. The project's core functionalities include querying and fetching data from MongoDB using Python, processing and analyzing the data using pandas, and providing useful insights. The value proposition of the project lies in its ability to streamline data retrieval and analysis tasks, making it easier to extract meaningful information from MongoDB databases.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | The project's architecture is not specified in the provided information. |
| 🔩 | **Code Quality**  | The code quality and style are not mentioned in the provided information. |
| 📄 | **Documentation** | The extent and quality of documentation are not specified in the provided information. |
| 🔌 | **Integrations**  | The key integrations and external dependencies of the project include: typing-inspect, telethon, pygooglenews, dotenv, marshmallow, starlette, pandas, beautifulsoup4, uvicorn, bertopic, openai, pymysql, SQLAlchemy, dnspython, numpy, sklearn, certifi, aiohttp, PyYAML, and urllib3. |
| 🧩 | **Modularity**    | The modularity and reusability of the codebase are not discussed in the provided information. |
| 🧪 | **Testing**       | No information is provided about the testing frameworks and tools used in the project. |
| ⚡️  | **Performance**   | The efficiency, speed, and resource usage of the project are not evaluated in the provided information. |
| 🛡️ | **Security**      | The measures used for data protection and access control are not discussed in the provided information. |
| 📦 | **Dependencies**  | The key external libraries and dependencies of the project include: typing-inspect, telethon, pygooglenews, dotenv, marshmallow, starlette, pandas, beautifulsoup4, uvicorn, bertopic, openai, pymysql, SQLAlchemy, dnspython, numpy, sklearn, certifi, aiohttp, PyYAML, and urllib3. |


---

##  Repository Structure

```sh
└── ./
    ├── package-lock.json
    ├── package.json
    ├── requirements.txt
    └── src
        ├── database
        │   ├── database_example.py
        │   └── processScrapeTelegram.py
        ├── frontend
        │   └── streamlit
        ├── helper
        │   ├── scraping
        │   └── textcleaner
        ├── machine_learning
        │   ├── BERTopic
        │   ├── NER
        │   ├── chat
        │   └── sentiment
        └── pipeline
            ├── 1_predictTopicLabel.py
            ├── 2_calculateMessageWithoutBertTopic.py
            ├── 3_assignEmbeddingToMessage.py
            ├── 3_updateTopicFrequencyCount.py
            ├── 4_addCompletionLabel.py
            └── requirements.txt
```

---

##  Modules

<details closed><summary>.</summary>

| File                                   | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---                                    | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [requirements.txt](requirements.txt)   | The code snippet in the pipeline directory contains a series of Python scripts that automate the processing and analysis of text messages. It involves predicting topic labels, calculating message metrics, assigning embeddings, updating topic frequencies, and adding completion labels. The snippet contributes to the overall architecture of the parent repository by providing a streamlined workflow for text analysis and data manipulation. |
| [package-lock.json](package-lock.json) | The `package-lock.json` file in the repository is used to track and manage the dependencies of the project. It ensures that all team members are working with the same versions of the required packages and libraries. It plays a critical role in maintaining consistency and reliability in the software development process.                                                                                                                       |
| [package.json](package.json)           | The `package.json` file in the parent repository's structure specifies the dependencies required for the project. It includes the `cors` and `dotenv` packages, which are essential for enabling cross-origin resource sharing and managing environment variables, respectively.                                                                                                                                                                       |

</details>

<details closed><summary>src.database</summary>

| File                                                              | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---                                                               | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [database_example.py](src/database/database_example.py)           | The code snippet in `database_example.py` connects to a MongoDB database, queries data based on specified conditions, and performs operations such as sorting, removing fields, and updating fields in the database. It fetches data from the `telegram_sample` collection and provides a self-designed interface to access the database.                                                                                                                     |
| [processScrapeTelegram.py](src/database/processScrapeTelegram.py) | This code snippet, located in the `src/database/processScrapeTelegram.py` file, contains functions that perform various operations on a MongoDB collection. These operations include calculating message training requirements, checking for redundant and missing embeddings, adding and clearing embeddings, updating field names, and more. The functions utilize the `pymongo` library to interact with the database and perform data manipulation tasks. |

</details>

<details closed><summary>src.pipeline</summary>

| File                                                                                        | Summary                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---                                                                                         | ---                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [3_updateTopicFrequencyCount.py](src/pipeline/3_updateTopicFrequencyCount.py)               | The code snippet in `3_updateTopicFrequencyCount.py` aggregates and updates the count of predicted class occurrences in a MongoDB database. It performs grouping and sorting operations based on messageDate, state, country, and predicted_class. The results are stored in a specified output database and collection.                                                                                                     |
| [requirements.txt](src/pipeline/requirements.txt)                                           | The code snippet in `src/pipeline/requirements.txt` lists the required packages for the pipeline module in the repository. It ensures the availability of key libraries such as argparse, pandas, and pymongo, enabling the pipeline scripts to run successfully.                                                                                                                                                            |
| [4_addCompletionLabel.py](src/pipeline/4_addCompletionLabel.py)                             | The code snippet in src/pipeline/4_addCompletionLabel.py connects to a MongoDB database and updates documents in a collection. It adds a topicUpdateDate field to new data that meets certain criteria. This code is part of a larger pipeline that processes and labels messages.                                                                                                                                           |
| [2_calculateMessageWithoutBertTopic.py](src/pipeline/2_calculateMessageWithoutBertTopic.py) | This code snippet, located at `src/pipeline/2_calculateMessageWithoutBertTopic.py`, calculates the number of new messages in a database collection that still need to be labeled with a topic. It uses MongoDB aggregation to filter the relevant data and outputs the results to the console. The script can be run with command-line arguments to specify the input database.                                              |
| [3_assignEmbeddingToMessage.py](src/pipeline/3_assignEmbeddingToMessage.py)                 | This code snippet, located at src/pipeline/3_assignEmbeddingToMessage.py, is responsible for assigning embeddings to messages in a MongoDB collection. It uses the OpenAI API to generate embeddings for text data that meets certain requirements. The embeddings are then added to the corresponding documents in the collection. The code can be executed by specifying the output database using command-line arguments. |
| [1_predictTopicLabel.py](src/pipeline/1_predictTopicLabel.py)                               | The code snippet plays a critical role in the parent repository's architecture. It achieves various tasks, including database processing, frontend development, helper functions, machine learning tasks, and pipeline management. It is a vital component in the overall functioning of the repository.                                                                                                                     |

</details>

<details closed><summary>src.frontend.streamlit</summary>

| File                                                      | Summary                                                                                                                                                                                                                                                                                                                |
| ---                                                       | ---                                                                                                                                                                                                                                                                                                                    |
| [app_EU_multi.py](src/frontend/streamlit/app_EU_multi.py) | The code snippet in `src/frontend/streamlit/app_EU_multi.py` is part of the parent repository's architecture. It imports necessary libraries and configures the page layout for a Streamlit app. Its main role is to create an interactive web application with various data visualization features related to the EU. |

</details>

<details closed><summary>src.machine_learning.NER</summary>

| File                                                                                                                      | Summary                                                                                                                                                                                                                                                                                          |
| ---                                                                                                                       | ---                                                                                                                                                                                                                                                                                              |
| [davlan-bert-base-multilingual-cased-ner-hrl.py](src/machine_learning/NER/davlan-bert-base-multilingual-cased-ner-hrl.py) | This code snippet represents the Davlan Bert Base Multilingual Cased NER HRL model. It is responsible for running Named Entity Recognition (NER) on different types of data sources such as Telegram, Twitter, and Google News. The code utilizes the transformers library to perform NER tasks. |

</details>

<details closed><summary>src.machine_learning.chat</summary>

| File                                                                                       | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---                                                                                        | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [Chatbot_test_edition.py](src/machine_learning/chat/Chatbot_test_edition.py)               | The code snippet in `Chatbot_test_edition.py` is responsible for querying a MongoDB Atlas database using specified parameters, retrieving relevant documents, and generating a response using a conversational retrieval chatbot. The code handles authentication, establishes a connection to the database, sets up embeddings for text search, defines memory for chat history, and configures the retrieval chain. Finally, it executes the query, retrieves the relevant information from the retrieved documents, and prints the answer. |
| [Chatbot_development_edition.py](src/machine_learning/chat/Chatbot_development_edition.py) | This code snippet contains a FastAPI endpoint that interacts with a chatbot. It connects to MongoDB Atlas, utilizes OpenAI language models, and performs conversational retrieval based on user queries. The snippet also handles parsing parameters, creating a chatbot chain, and returning relevant metadata.                                                                                                                                                                                                                              |
| [Chatbot_tutorial_edition.py](src/machine_learning/chat/Chatbot_tutorial_edition.py)       | The code snippet in `Chatbot_tutorial_edition.py` implements a chatbot using Streamlit framework. It integrates with MongoDB, loads CSV data, and uses OpenAI models for conversational retrieval. The chatbot accepts user queries and generates responses based on the chat history.                                                                                                                                                                                                                                                        |
| [chat_fastapi_app.py](src/machine_learning/chat/chat_fastapi_app.py)                       | This code snippet is a FastAPI endpoint that processes user queries and returns relevant answers. It parses input parameters, constructs search conditions for MongoDB Atlas VectorSearch, initializes MongoDB connection, sets up embeddings, vectors, and memory for the retrieval chain, and processes the query using the retrieval chain. It also adds the new Q&A pair to the chat history.                                                                                                                                             |

</details>

<details closed><summary>src.machine_learning.chat.docker</summary>

| File                                                                                              | Summary                                                                                                                                                                                                                                                                                                                                     |
| ---                                                                                               | ---                                                                                                                                                                                                                                                                                                                                         |
| [docker-compose.yaml](src/machine_learning/chat/docker/docker-compose.yaml)                       | The code snippet in `src/machine_learning/chat/docker/docker-compose.yaml` defines a Docker Compose configuration file that builds and runs a chat application using FastAPI. It sets environment variables for authentication and networking.                                                                                              |
| [requirements.txt](src/machine_learning/chat/docker/requirements.txt)                             | The code snippet located at `src/machine_learning/chat/docker/requirements.txt` contains the necessary dependencies required for running the chat module of the software. These dependencies include libraries for handling HTTP requests, data serialization, and various utilities.                                                       |
| [chat_fastapi_conversational.py](src/machine_learning/chat/docker/chat_fastapi_conversational.py) | This code snippet represents a FastAPI web service for a chatbot. It handles POST requests to /query, processes parameters, connects to MongoDB, generates a conversation chain, and returns an answer. It also handles GET requests to /test.                                                                                              |
| [chat_fastapi.py](src/machine_learning/chat/docker/chat_fastapi.py)                               | The code snippet `chat_fastapi.py` is part of the machine learning module in the repository's architecture. It provides a FastAPI endpoint for querying a chatbot that uses a language model and retrieval-based QA. The endpoint accepts query parameters and returns chatbot responses based on the provided parameters and chat history. |
| [chat_fastapi.dockerfile](src/machine_learning/chat/docker/chat_fastapi.dockerfile)               | This code snippet is a Dockerfile that sets up an image for running a FastAPI chat service. It installs dependencies, sets the working directory, and specifies the command to run the service.                                                                                                                                             |
| [.env](src/machine_learning/chat/docker/.env)                                                     | This code snippet, located in the `src/machine_learning/chat/docker/.env` file, contains environment variables related to authentication tokens and API keys. It provides secure access to external services used in the parent repository's machine learning chat module.                                                                  |

</details>

<details closed><summary>src.machine_learning.BERTopic</summary>

| File                                                                                   | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---                                                                                    | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [requirements.txt](src/machine_learning/BERTopic/requirements.txt)                     | The code snippet in `src/machine_learning/BERTopic/requirements.txt` specifies the required dependencies for the BERTopic module. It enables topic modeling, classification, and clustering of text data using BERT embeddings and Python libraries such as sklearn, pandas, and regex. The code achieves efficient and accurate text analysis without delving into its technical implementation details.                                                                      |
| [testing.ipynb](src/machine_learning/BERTopic/testing.ipynb)                           | The code snippet is part of a repository's database module. It contains a Python script that demonstrates database functionality.                                                                                                                                                                                                                                                                                                                                              |
| [runBERTopic.py](src/machine_learning/BERTopic/runBERTopic.py)                         | This code snippet is a part of a larger repository with a multi-lingual BERTopic model. The snippet provides functions to train the BERTopic model on different types of data sources such as Telegram, Twitter, and Google News. It also includes functions to save and load the model and to perform inference on the data. The code incorporates data preprocessing steps, topic modeling, visualization, and saving of results.                                            |
| [preprocess_testing_data.py](src/machine_learning/BERTopic/preprocess_testing_data.py) | This code snippet is responsible for organizing and processing data for future use of semi-supervised BERTopic. It loads a CSV file, maps cluster names to cluster IDs, fetches data from MongoDB, applies data preprocessing techniques, and saves the processed dataset to a new CSV file.                                                                                                                                                                                   |
| [upload_model.py](src/machine_learning/BERTopic/upload_model.py)                       | This code snippet in `upload_model.py` uploads a pre-trained BERTopic model and associated data to the HuggingFace Hub. It also reads a CSV file, creates a dictionary, and prints the dictionary's contents. This code contributes to the machine learning functionality of the repository by making the trained model available for others to use.                                                                                                                           |
| [mapping.py](src/machine_learning/BERTopic/mapping.py)                                 | This code snippet maps topics from a text document to predefined categories using a dictionary. It then connects to a MongoDB database and updates the category for each document based on the matching keywords. The main role of this code is to facilitate topic categorization and database operations in the parent repository's architecture.                                                                                                                            |
| [apply_BERTopic_mongoDB.py](src/machine_learning/BERTopic/apply_BERTopic_mongoDB.py)   | The code snippet `apply_BERTopic_mongoDB.py` in the `machine_learning/BERTopic` directory of the repository applies BERTopic on text data fetched from a MongoDB database. It predicts topics using a pre-trained model and updates the records with the predicted classes in parallel.                                                                                                                                                                                        |
| [runsemi-supervised-bert.py](src/machine_learning/BERTopic/runsemi-supervised-bert.py) | This code snippet is the implementation of a BERTopic model in the machine_learning module of a larger codebase. It fits the model, visualizes the results, and saves the representative documents and model to disk. The code utilizes UMAP and HDBSCAN for dimension reduction and clustering, respectively. It also employs CountVectorizer for preprocessing the documents. The results are saved as HTML files and an Excel file containing the representative documents. |
| [validation.py](src/machine_learning/BERTopic/validation.py)                           | This code snippet is part of the parent repository's architecture and is responsible for managing the database, frontend, helper functions, machine learning models, and pipeline processes. It performs various tasks such as data processing, scraping, text cleaning, topic prediction, sentiment analysis, and more.                                                                                                                                                       |

</details>

<details closed><summary>src.machine_learning.sentiment</summary>

| File                                                                                                                                | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---                                                                                                                                 | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [sentiment.py](src/machine_learning/sentiment/sentiment.py)                                                                         | This code snippet is part of a larger repository structure. It implements a sentiment analysis classifier using XLM-Roberta and RoBERTa models. The classifier can predict the sentiment and emotion of text data. It takes an input file, performs inference on the data, and outputs the results to a CSV file. The code also includes options to set the device for GPU or CPU usage.                                                                                                                          |
| [cardiffnlp-twitter-xlm-roberta-base-sentiment.py](src/machine_learning/sentiment/cardiffnlp-twitter-xlm-roberta-base-sentiment.py) | This code snippet is a sentiment analysis model using CardiffNLP's Twitter RoBERTa base pre-trained model. It classifies sentiment for data from Telegram, Twitter, and Google News sources. The sentiment analysis is performed using a pipeline that tokenizes the text and predicts the sentiment label. The results are saved in a CSV file. This code snippet is part of a larger repository with a directory structure that includes database, frontend, helper, machine_learning, and pipeline components. |
| [cardiffnlp-twitter-roberta-base-emotion.py](src/machine_learning/sentiment/cardiffnlp-twitter-roberta-base-emotion.py)             | This code snippet is responsible for running the CardiffBLP Twitter RoBERTa base emotion model. It classifies emotions in text data from different sources such as Telegram, Twitter, and Google News. The results are saved in CSV files.                                                                                                                                                                                                                                                                        |

</details>

<details closed><summary>src.helper.scraping.telegram_tools</summary>

| File                                                                                                    | Summary                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---                                                                                                     | ---                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [requirements.txt](src/helper/scraping/telegram_tools/requirements.txt)                                 | This code snippet, located in `src/helper/scraping/telegram_tools/requirements.txt`, lists the required external dependencies for scraping data from Telegram. The dependencies include libraries for argument parsing, asynchronous programming, date/time handling, geocoding, data manipulation, database interaction, environment variables, and progress tracking.                                                        |
| [extractSateAndCity.py](src/helper/scraping/telegram_tools/extractSateAndCity.py)                       | This code snippet is responsible for parsing the state and city information for each chat. It can store the output in a local file or a MongoDB database. The code uses country-state and state-city mappings to extract the relevant information from the input file. The output can be written to a local file or a specified database collection.                                                                           |
| [scrapeTelegramChannelMessages.py](src/helper/scraping/telegram_tools/scrapeTelegramChannelMessages.py) | The `scrapeTelegramChannelMessages.py` code snippet is responsible for scraping message text and metadata from Telegram channels. It utilizes the Telethon library to connect to the Telegram API, retrieves messages from specified channels, and saves the scraped data to a MongoDB database. It also includes functionality to parse state and city information from channel names and obtain embeddings for each message. |
| [generateTelegramStringToken.py](src/helper/scraping/telegram_tools/generateTelegramStringToken.py)     | The code snippet generates a Telegram string token using the Telethon library. It authenticates with the Telegram API using the provided API ID and API hash and saves the generated token. This token is necessary for executing other scripts in the repository that interact with the Telegram API.                                                                                                                         |

</details>

<details closed><summary>src.helper.scraping.twitter_tools</summary>

| File                                                                               | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                                                                                | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [count_tweets_single.sh](src/helper/scraping/twitter_tools/count_tweets_single.sh) | This code snippet, located at `src/helper/scraping/twitter_tools/count_tweets_single.sh`, counts the number of tweets that match a specific query and time range using the Twitter API. It uses a YAML file for authentication and outputs the result to a JSON file. The script then extracts the tweet count from the JSON file and prints the total count.                                                                                                                                                                                                                         |
| [scrape_tweets.ipynb](src/helper/scraping/twitter_tools/scrape_tweets.ipynb)       | This code snippet contributes to the parent repository's architecture by implementing critical features related to database management, frontend development, helper functions, and machine learning components. The code achieves tasks such as processing scraped data, cleaning text, predicting topic labels, calculating message embeddings, and updating topic frequency counts.                                                                                                                                                                                                |
| [search_tweets.sh](src/helper/scraping/twitter_tools/search_tweets.sh)             | This code snippet, located at `src/helper/scraping/twitter_tools/search_tweets.sh`, performs data scraping from Twitter based on specified queries. It uses a query file and outputs the results to a JSON file. The script also has an optional step to create a CSV file from the JSON data.                                                                                                                                                                                                                                                                                        |
| [count_tweets.sh](src/helper/scraping/twitter_tools/count_tweets.sh)               | This code snippet, located at src/helper/scraping/twitter_tools/count_tweets.sh, is responsible for counting the number of tweets based on specific queries. It iterates through the provided queries, searches for tweets using the Twitter API, and calculates the total count of tweets found. The result is printed at the end of the execution.                                                                                                                                                                                                                                  |
| [search_tweets.py](src/helper/scraping/twitter_tools/search_tweets.py)             | This code snippet, located at `src/helper/scraping/twitter_tools/search_tweets.py`, is responsible for searching and retrieving tweets from Twitter based on specified parameters such as query, start time, end time, etc. It utilizes the `searchtweets` library and allows for command-line configuration via various arguments. The retrieved tweets can be printed to stdout or saved in JSON format.                                                                                                                                                                            |
| [create_tweets_csv.py](src/helper/scraping/twitter_tools/create_tweets_csv.py)     | This code snippet, located at `src/helper/scraping/twitter_tools/create_tweets_csv.py`, takes a query file in JSON format as input and processes the data to create a CSV file. It extracts relevant information from the JSON file, such as tweet text and creation date, and writes it to the CSV file. The resulting CSV file contains the collected tweets, and the script also prints the number of tweets and the number of unique tweets. This code snippet is responsible for collecting and organizing Twitter data for further analysis within the repository architecture. |

</details>

<details closed><summary>src.helper.scraping.traditional_news_tools.google</summary>

| File                                                                                                                           | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                            | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [googlescapper.py](src/helper/scraping/traditional_news_tools/google/googlescapper.py)                                         | This code snippet is a web scraper that fetches news articles using Google News API, processes the content, and saves it to a MongoDB database. It supports multi-threading for efficient scraping and includes error handling. The code is part of a larger repository structure that encompasses other components like database handling, machine learning, and frontend development.                                                                                                         |
| [scrapeGoogleNewsPageLinksSoup.py](src/helper/scraping/traditional_news_tools/google/scrapeGoogleNewsPageLinksSoup.py)         | This code snippet, located in src/helper/scraping/traditional_news_tools/google/scrapeGoogleNewsPageLinksSoup.py, scrapes Google News articles and saves them as text files. It checks existing files to avoid duplication and uses multithreading for efficiency.                                                                                                                                                                                                                              |
| [scrapeGoogleNews.py](src/helper/scraping/traditional_news_tools/google/scrapeGoogleNews.py)                                   | This code snippet is part of a larger codebase with a directory structure. The snippet, located at `src/helper/scraping/traditional_news_tools/google/scrapeGoogleNews.py`, implements a web scraping function to extract news articles from Google News. It uses parallel execution to perform multiple searches simultaneously and saves the results to a CSV file. The snippet plays a critical role in collecting news data for further processing in the parent repository's architecture. |
| [preprocessMediaData.py](src/helper/scraping/traditional_news_tools/google/preprocessMediaData.py)                             | This code snippet, located at `src/helper/scraping/traditional_news_tools/google/preprocessMediaData.py`, is responsible for preprocessing media data from Google News articles. It filters out articles that contain specific keywords related to migration and refugees. It further removes lines that are too short, do not start with any character, or do not match the main text pattern. The filtered articles are saved in a separate directory.                                        |
| [scrapeGoogleNewsPageLinksSelenium.py](src/helper/scraping/traditional_news_tools/google/scrapeGoogleNewsPageLinksSelenium.py) | This code snippet is responsible for scraping and saving article content from Google News pages using Selenium. It utilizes threading to process multiple URLs concurrently, improving efficiency. The scraped content is then saved to text files for further analysis.                                                                                                                                                                                                                        |

</details>

<details closed><summary>src.helper.scraping.traditional_news_tools.GDELT</summary>

| File                                                                                  | Summary                                                                                                                                                                                                                                                                     |
| ---                                                                                   | ---                                                                                                                                                                                                                                                                         |
| [scrapingGDELT.py](src/helper/scraping/traditional_news_tools/GDELT/scrapingGDELT.py) | This code snippet queries the GDELT dataset stored in Google BigQuery to retrieve events that occurred in the US between February 1, 2023, and February 28, 2023. The results are then printed. The code plays a critical role in scraping GDELT data for further analysis. |

</details>

<details closed><summary>src.helper.textcleaner</summary>

| File                                            | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                                             | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [cleaner.py](src/helper/textcleaner/cleaner.py) | This code snippet is part of a larger repository structure and is located in the file `src/helper/textcleaner/cleaner.py`. It contains a `Cleaner` class with methods for cleaning multilingual text, dropping sequences without words, correcting spelling mistakes, removing emojis, and cleaning news articles, Telegram messages, and Twitter data. The code takes input arguments, processes the data accordingly, and saves the cleaned output. |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version x.y.z`

###  Installation

1. Clone the . repository:

```sh
git clone ../.
```

2. Change to the project directory:

```sh
cd .
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

###  Running .

Use the following command to run .:

```sh
python main.py
```

###  Tests

To execute tests, run:

```sh
pytest
```

---

##  Project Roadmap

- [X] `► INSERT-TASK-1`
- [ ] `► INSERT-TASK-2`
- [ ] `► ...`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://local//blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://local//discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://local//issues)**: Submit bugs found or log feature requests for ..

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone ../.
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-quick-links)

---
