<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
</p>
<p align="center">
    <h1 align="center">R2G</h1>
</p>
<p align="center">
    <em>Streamline your data analysis with R2G</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/GIQSubmission/R2G?style=default&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/GIQSubmission/R2G?style=default&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/GIQSubmission/R2G?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/GIQSubmission/R2G?style=default&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Features](#-features)
> - [ Repository Structure](#-repository-structure)
> - [ Modules](#-modules)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [ Running R2G](#-running-R2G)
>   - [ Tests](#-tests)
> - [ Project Roadmap](#-project-roadmap)
> - [ Contributing](#-contributing)
> - [ License](#-license)
> - [ Acknowledgments](#-acknowledgments)

---

##  Overview

R2G is a project that offers a powerful and efficient solution for web scraping and data retrieval tasks. It provides a wide range of functionalities and tools to facilitate the extraction, processing, and storage of data from various sources. R2G's value proposition lies in its ability to automate the collection of data from websites and APIs, enabling users to access and analyze relevant information without the need for manual intervention. With support for popular libraries like BeautifulSoup and PyGoogleNews, R2G simplifies the scraping process and empowers users to extract valuable insights from the web. Whether it's gathering news articles, monitoring social media, or aggregating data for research purposes, R2G streamlines the data retrieval process and enhances productivity.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | The project's architecture is not described in the repository. Further analysis would be needed to determine the architecture. |
| 🔩 | **Code Quality**  | The code quality and style are not mentioned in the repository. Without further information or code review, it is difficult to assess the code quality. |
| 📄 | **Documentation** | There is no explicit information about the extent and quality of documentation in the repository. It is recommended to review the repository and codebase to evaluate the documentation. |
| 🔌 | **Integrations**  | The key integrations and external dependencies are mentioned in the project dependencies list. Some notable dependencies include Vue.js, FastAPI, SQLAlchemy, pymongo, and leaflet. |
| 🧩 | **Modularity**    | The modularity and reusability of the codebase are not explicitly discussed in the repository. A code review would be necessary to evaluate the modularity. |
| 🧪 | **Testing**       | No specific testing frameworks or tools are mentioned in the repository. It is recommended to review the codebase to determine the testing approach. |
| ⚡️  | **Performance**   | There is no information available in the repository regarding the efficiency, speed, and resource usage of the project. In-depth analysis would be required to assess performance. |
| 🛡️ | **Security**      | The security measures used for data protection and access control are not explicitly mentioned in the repository. Further investigation would be necessary to evaluate the security implementation. |
| 📦 | **Dependencies**  | The key external libraries and dependencies are provided in the project dependencies list. Some notable dependencies include tenacity, ipynb, element-plus, echarts, and axios.|
| 🚀 | **Scalability**   | The ability of the project to handle increased traffic and load is not discussed in the repository. Further analysis would be required to assess scalability. |


---

##  Repository Structure

```sh
└── R2G/
    ├── frontend
    │   ├── chatgpt-backend
    │   │   ├── Tumen_Chatbot_development_edition.py
    │   │   ├── chatGpt.controller.js
    │   │   ├── chat_fastapi_app.py
    │   │   ├── index.js
    │   │   ├── langchain.js
    │   │   ├── package-lock.json
    │   │   └── package.json
    │   ├── mongoDB-backend
    │   │   ├── .env
    │   │   └── connect.js
    │   ├── package-lock.json
    │   ├── package.json
    │   └── r2g2_vue
    │       ├── _.headers
    │       ├── index.html
    │       ├── package-lock.json
    │       ├── package.json
    │       ├── src
    │       └── vite.config.js
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

| File                                                                                    | Summary                                                                                                                                                                                                                                                                                                                                                      |
| ---                                                                                     | ---                                                                                                                                                                                                                                                                                                                                                          |
| [requirements.txt](https://github.com/GIQSubmission/R2G/blob/master/requirements.txt)   | This code snippet, located in the frontend directory, is responsible for managing the backend logic of the chatGpt application. It includes files like Tumen_Chatbot_development_edition.py, chatGpt.controller.js, chat_fastapi_app.py, and langchain.js. These files handle the processing and communication between the frontend and the chatbot backend. |
| [.gitignore](https://github.com/GIQSubmission/R2G/blob/master/.gitignore)               | The code snippet is part of the R2G repository and is located in the frontend directory. It ignores the node_modules and frontend/node_modules folders when tracking changes in Git.                                                                                                                                                                         |
| [package-lock.json](https://github.com/GIQSubmission/R2G/blob/master/package-lock.json) | The `package-lock.json` file in the `R2G` repository stores the dependencies and their specific versions for the project. It ensures consistent builds across different development environments by locking down the exact versions of the dependencies required.                                                                                            |
| [package.json](https://github.com/GIQSubmission/R2G/blob/master/package.json)           | This code snippet, located in the package.json file, includes dependencies for the parent repository's architecture. It ensures compatibility and functionality by including packages for handling CORS and environment variables.                                                                                                                           |

</details>

<details closed><summary>frontend</summary>

| File                                                                                             | Summary                                                                                                                                                                                                                                          |
| ---                                                                                              | ---                                                                                                                                                                                                                                              |
| [package-lock.json](https://github.com/GIQSubmission/R2G/blob/master/frontend/package-lock.json) | The code snippet in the chatgpt-backend folder of the R2G repository is responsible for implementing the backend logic for a chatbot. It includes files for the chatbot's controller and server setup.                                           |
| [package.json](https://github.com/GIQSubmission/R2G/blob/master/frontend/package.json)           | The code snippet in frontend/package.json is responsible for managing the dependencies required by the frontend of the R2G repository. It ensures that the necessary packages, such as axios, express, and openai, are installed and up to date. |

</details>

<details closed><summary>frontend.mongoDB-backend</summary>

| File                                                                                               | Summary                                                                                                                                                                                                                                                                                                                      |
| ---                                                                                                | ---                                                                                                                                                                                                                                                                                                                          |
| [connect.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/mongoDB-backend/connect.js) | The code snippet in `connect.js` establishes a connection to a MongoDB database and creates an Express server. It handles requests to retrieve data from specific collections in the database. It also implements CORS to allow access from specific origins. The server runs on a specified port and returns health status. |
| [.env](https://github.com/GIQSubmission/R2G/blob/master/frontend/mongoDB-backend/.env)             | This code snippet in the `frontend/mongoDB-backend` directory of the repository's architecture manages the MongoDB connection using the provided URL.                                                                                                                                                                        |

</details>

<details closed><summary>frontend.chatgpt-backend</summary>

| File                                                                                                                                                   | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                                                    | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [chatGpt.controller.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/chatgpt-backend/chatGpt.controller.js)                               | The code snippet `chatGpt.controller.js` is a controller module that interacts with OpenAI's GPT-3.5 Turbo model for chat-based conversational AI. It creates an OpenAI instance, configures it with the provided API key, and sends a user's message to the model for generating a reply. The reply is then returned as a response.                                                                                                                                                                                                                                                                                            |
| [Tumen_Chatbot_development_edition.py](https://github.com/GIQSubmission/R2G/blob/master/frontend/chatgpt-backend/Tumen_Chatbot_development_edition.py) | This code snippet, located at `frontend/chatgpt-backend/Tumen_Chatbot_development_edition.py`, is responsible for creating a FastAPI web server that interacts with a chatbot. It processes user queries, retrieves relevant information from a MongoDB database, and generates responses using the OpenAI GPT-3.5 Turbo language model. The code handles filtering and parsing parameters, connects to the database, performs text embeddings and vector searches, and constructs a conversational retrieval chain. The server accepts POST requests to `/query` and returns the chatbot's answer along with the chat history. |
| [index.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/chatgpt-backend/index.js)                                                         | The `index.js` file in the `frontend/chatgpt-backend` directory acts as the server entry point for the chatbot functionality. It sets up an Express server, handles CORS and JSON parsing, and exposes a `/chatbot` endpoint for interacting with the chat GPT model.                                                                                                                                                                                                                                                                                                                                                           |
| [package-lock.json](https://github.com/GIQSubmission/R2G/blob/master/frontend/chatgpt-backend/package-lock.json)                                       | The code snippet in the chatgpt-backend folder of the R2G repository serves as the backend implementation for a chatbot built with ChatGPT. It includes files and logic for handling user input, generating responses, and handling language translation. It integrates with a MongoDB backend and utilizes FastAPI and JavaScript technologies.                                                                                                                                                                                                                                                                                |
| [package.json](https://github.com/GIQSubmission/R2G/blob/master/frontend/chatgpt-backend/package.json)                                                 | The code snippet in the file `frontend/chatgpt-backend/package.json` is responsible for managing the dependencies required for the backend of the ChatGPT application. It specifies the necessary packages like `body-parser`, `cors`, `dotenv`, `express`, `langchain`, and `openai` along with their respective versions.                                                                                                                                                                                                                                                                                                     |
| [langchain.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/chatgpt-backend/langchain.js)                                                 | The `langchain.js` code snippet in the `frontend/chatgpt-backend` directory is a server-side implementation of a chatbot API. It utilizes various NLP libraries like OpenAI, MongoDB, and Langchain to retrieve relevant documents and provide accurate answers to user questions. The API listens for POST requests and returns the chatbot's response.                                                                                                                                                                                                                                                                        |
| [chat_fastapi_app.py](https://github.com/GIQSubmission/R2G/blob/master/frontend/chatgpt-backend/chat_fastapi_app.py)                                   | This code snippet is a FastAPI backend application that handles user queries and returns relevant answers. It connects to MongoDB Atlas for data retrieval and utilizes various components from the langchain package, such as embeddings, conversation memory, and retrieval chains, to process the queries. The code also parses parameters, constructs search conditions, and integrates with the OpenAI API for generating answers.                                                                                                                                                                                         |

</details>

<details closed><summary>frontend.r2g2_vue</summary>

| File                                                                                                      | Summary                                                                                                                                                                                                                                                                   |
| ---                                                                                                       | ---                                                                                                                                                                                                                                                                       |
| [index.html](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/index.html)               | The code snippet in the `frontend/r2g2_vue/index.html` file is responsible for rendering the main Vue.js application in the browser. It includes the necessary HTML structure, sets the document title, and imports the main JavaScript module for the Vue.js app.        |
| [vite.config.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/vite.config.js)       | The `vite.config.js` file in the `frontend/r2g2_vue` directory is responsible for configuring the Vite build tool for the Vue.js frontend. It sets up the Vue plugin and resolves the alias for the `src` directory using the `fileURLToPath` and `URL` functions.        |
| [.gitignore](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/.gitignore)               | The code snippet in `frontend/r2g2_vue/.gitignore` specifies the files and directories that should be ignored by Git. It excludes log files, editor directories, and generated build artifacts, among others, from being tracked by version control.                      |
| [package-lock.json](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/package-lock.json) | The code snippet in the `chatgpt-backend` folder of the `frontend` section in the `R2G` repository is a critical component that enables chat functionality powered by the ChatGPT model. It includes a Python script, a JavaScript controller, and other necessary files. |
| [package.json](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/package.json)           | This code snippet contains the package.json file for the r2g2_vue frontend project. It specifies the project's dependencies, development dependencies, and scripts for building and running the project.                                                                  |
| [_.headers](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/_.headers)                 | The code snippet in frontend/r2g2_vue/_.headers sets content security policies to restrict sources for images, scripts, and styles. It enhances the security of the parent repository's frontend by preventing potential security vulnerabilities and attacks.            |

</details>

<details closed><summary>frontend.r2g2_vue.src</summary>

| File                                                                                                              | Summary                                                                                                                                                                                                                                                                                                                                                |
| ---                                                                                                               | ---                                                                                                                                                                                                                                                                                                                                                    |
| [App.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/App.vue)                         | This code snippet represents the main Vue.js component of the R2G repository's frontend structure. It provides the basic template and structure for rendering the application's views.                                                                                                                                                                 |
| [plugins.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/plugins.js)                   | The code snippet in `frontend/r2g2_vue/src/plugins.js` is a plugin file that defines several functions used in the R2G repository. These functions handle CSV data, change the language, get cluster and state categories, count clusters, apply filters, and convert timestamps. The file enhances the functionality of the R2G frontend application. |
| [myplugins.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/myplugins.js)               | The myplugins.js code snippet in the r2g2_vue component of the repository provides functions for manipulating and analyzing data related to telecommunication clusters. It enables changing locale, getting cluster and state categories, retrieving and filtering dates, counting clusters, converting timestamps, and calculating moving averages.   |
| [main.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/main.js)                         | This code snippet initializes a Vue.js application by creating an app, setting up internationalization, configuring Axios for HTTP requests, integrating Element Plus UI library, and mounting the app to a specific DOM element. It also includes plugins and registers a Loading component.                                                          |
| [plugins4telegram.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/plugins4telegram.js) | The code snippet is part of the R2G repository's frontend architecture. It defines various functions related to data manipulation for a Telegram chat application, including language changes, retrieving cluster and state categories, filtering data based on country and state, and performing moving average calculations.                         |

</details>

<details closed><summary>frontend.r2g2_vue.src.components</summary>

| File                                                                                                                   | Summary                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---                                                                                                                    | ---                                                                                                                                                                                                                                                                                                                                                                                                     |
| [DatePicker.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/components/DatePicker.vue)     | This code snippet is a Vue.js component called DatePicker.vue located in the frontend/r2g2_vue/src/components directory of the repository. It provides a date picker with start and end date selection, formatted display, and disabled date functionality. It emits the selected date range to the parent component.                                                                                   |
| [Telegram.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/components/Telegram.vue)         | The code snippet is part of the R2G repository and is located in the frontend/chatgpt-backend directory. It includes files related to a chatbot using ChatGPT, FastAPI, and MongoDB. The main role of this code is to handle chatbot functionalities, such as processing requests and providing responses.                                                                                              |
| [Home.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/components/Home.vue)                 | The code snippet represents the Home.vue component in the frontend of the R2G repository. It handles user input for selecting topics, countries, states, and date range, and displays a line chart and a chatbot based on the selected options.                                                                                                                                                         |
| [MapComponent.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/components/MapComponent.vue) | The MapComponent.vue code snippet is responsible for rendering a map component using `vue-leaflet` library. It displays a map with features based on geojson data and allows zooming, centering, and highlighting of states. The component also shows tooltips when hovering over states and emits selected state events.                                                                               |
| [Other.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/components/Other.vue)               | The code snippet in `Other.vue` is a Vue.js component that represents a user interface for selecting various options related to the Ukrainian Refugee Crisis. It includes features like selecting a country, state, and date range of interest, displaying a map, line chart, and a chatbot. The component communicates with other components to update the selected options and display relevant data. |
| [ChatBot.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/components/ChatBot.vue)           | The `ChatBot.vue` component in the `frontend/r2g2_vue/src/components` directory of the parent repository is responsible for displaying a chat bot interface and handling user interactions. Users can ask questions using predefined templates or type their own questions, and the component sends those questions to a chatbot server and displays the responses in the chat interface.               |
| [Navigation.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/components/Navigation.vue)     | This code snippet is a Vue component called Navigation.vue located in the frontend/r2g2_vue/src/components/ directory. It represents a navigation menu with various options such as Home, Telegram, and language selection. It also includes a switch for toggling between dark and light mode.                                                                                                         |
| [LineChart.vue](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/components/LineChart.vue)       | The code snippet is a Vue component called LineChart.vue that renders a line chart using the Vue Chart.js library. It receives chart data and options as props and includes features such as responsive layout, time-based x-axis, and filtering based on selected state.                                                                                                                               |

</details>

<details closed><summary>frontend.r2g2_vue.src.router</summary>

| File                                                                                                   | Summary                                                                                                                                                                                                                                                                                      |
| ---                                                                                                    | ---                                                                                                                                                                                                                                                                                          |
| [request.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/router/request.js) | This code snippet defines an axios instance for sending HTTP requests in the frontend of the R2G repository. It includes error handling for different status codes and interceptors for request and response handling. The axios instance is exported for usage in other modules/components. |
| [router.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/router/router.js)   | This code snippet is the router module for the Vue frontend of the R2G repository. It creates routes for the Home and Telegram components, allowing navigation between them.                                                                                                                 |
| [path.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/router/path.js)       | The code snippet in `frontend/r2g2_vue/src/router/path.js` provides the base URL paths for various endpoints used in the frontend of the repository. It includes paths for accessing data files and interacting with a MongoDB server.                                                       |
| [api.js](https://github.com/GIQSubmission/R2G/blob/master/frontend/r2g2_vue/src/router/api.js)         | This code snippet defines an API module for making HTTP requests to the backend. It provides methods for fetching data from a MongoDB cluster and handling errors.                                                                                                                           |

</details>

<details closed><summary>src.database</summary>

| File                                                                                                               | Summary                                                                                                                                                                                                                                                                                                                                                            |
| ---                                                                                                                | ---                                                                                                                                                                                                                                                                                                                                                                |
| [database_example.py](https://github.com/GIQSubmission/R2G/blob/master/src/database/database_example.py)           | The code snippet at `src/database/database_example.py` connects to a MongoDB database, performs queries, updates and removes fields in the collection. It uses the `pymongo` library and returns data in a pandas DataFrame.                                                                                                                                       |
| [processScrapeTelegram.py](https://github.com/GIQSubmission/R2G/blob/master/src/database/processScrapeTelegram.py) | The code snippet located at `src/database/processScrapeTelegram.py` is responsible for performing various operations on a MongoDB collection. It includes functions to calculate message statistics, add and clear message embeddings, update field names, and manipulate collection data. The code interacts with the MongoDB database using the PyMongo library. |

</details>

<details closed><summary>src.pipeline</summary>

| File                                                                                                                                         | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---                                                                                                                                          | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [3_updateTopicFrequencyCount.py](https://github.com/GIQSubmission/R2G/blob/master/src/pipeline/3_updateTopicFrequencyCount.py)               | The code snippet located at src/pipeline/3_updateTopicFrequencyCount.py is responsible for aggregating message data from a MongoDB collection. It counts occurrences of predicted_class for each combination of messageDate, state, and country, and updates the count in an output database. The script connects to the MongoDB cluster, defines the aggregation pipeline, and executes the aggregation queries.                                                                         |
| [requirements.txt](https://github.com/GIQSubmission/R2G/blob/master/src/pipeline/requirements.txt)                                           | The code snippet in the `requirements.txt` file located at `src/pipeline/` defines the required Python packages for the pipeline component of the parent repository. These packages include asyncio, pandas, pymongo, and others.                                                                                                                                                                                                                                                         |
| [4_addCompletionLabel.py](https://github.com/GIQSubmission/R2G/blob/master/src/pipeline/4_addCompletionLabel.py)                             | The `4_addCompletionLabel.py` code snippet in the `src/pipeline` directory of the repository adds completion labels to new data in the specified MongoDB collection. It updates the topic update date of the data and assigns completion labels using a pretrained BERT model. The code connects to the MongoDB cluster, selects the new data, and updates the collection accordingly.                                                                                                    |
| [2_calculateMessageWithoutBertTopic.py](https://github.com/GIQSubmission/R2G/blob/master/src/pipeline/2_calculateMessageWithoutBertTopic.py) | The code in `2_calculateMessageWithoutBertTopic.py` calculates the number of new messages in a MongoDB collection that require topic labels from BERT. It retrieves the data and displays the count. The code is part of the R2G project's pipeline for processing scraped Telegram data.                                                                                                                                                                                                 |
| [3_assignEmbeddingToMessage.py](https://github.com/GIQSubmission/R2G/blob/master/src/pipeline/3_assignEmbeddingToMessage.py)                 | This code snippet, located at `src/pipeline/3_assignEmbeddingToMessage.py`, is responsible for assigning embeddings to messages in a MongoDB collection. It uses OpenAI's text-embedding-ada-002 model to generate embeddings. The code filters messages based on certain criteria and updates their embeddings in the collection.                                                                                                                                                        |
| [1_predictTopicLabel.py](https://github.com/GIQSubmission/R2G/blob/master/src/pipeline/1_predictTopicLabel.py)                               | The code snippet is part of the R2G repository and is located in the frontend/chatgpt-backend directory. It includes files like Tumen_Chatbot_development_edition.py and chatGpt.controller.js. The code serves as the backend for the ChatGpt module and implements functionalities related to language processing and chatbot interaction. It contributes to the architecture of the parent repository by providing the necessary backend support for the frontend chatbot application. |

</details>

<details closed><summary>src.frontend.streamlit</summary>

| File                                                                                                       | Summary                                                                                                                                                                          |
| ---                                                                                                        | ---                                                                                                                                                                              |
| [app_EU_multi.py](https://github.com/GIQSubmission/R2G/blob/master/src/frontend/streamlit/app_EU_multi.py) | This code snippet includes the backend implementation of a chatbot in the R2G repository. It is responsible for handling chat requests and communication with the chatGpt model. |

</details>

<details closed><summary>src.machine_learning.NER</summary>

| File                                                                                                                                                                       | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                                                                                                                                                                        | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [davlan-bert-base-multilingual-cased-ner-hrl.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/NER/davlan-bert-base-multilingual-cased-ner-hrl.py) | This code snippet is a part of the NER (Named Entity Recognition) module in the machine learning component of the R2G repository. It runs a Davlan Bert base multilingual cased NER HRL model on different data sources like Telegram, Twitter, and Google News. It preprocesses the text by removing smileys, punctuation, hashtags, and mentions before performing NER classification. The output is saved in a CSV file. The code also sets the device for GPU usage if available. |

</details>

<details closed><summary>src.machine_learning.chat</summary>

| File                                                                                                                                        | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---                                                                                                                                         | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [Chatbot_test_edition.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/Chatbot_test_edition.py)               | The code snippet is responsible for querying a MongoDB database using specified parameters such as start and end dates, country, state, predicted class, and a query string. It fetches the relevant data from the database and uses it to generate an answer using a conversational retrieval chain model. The generated answer includes metadata like state, country, message date, predicted class, and the actual response to the query.                                                                                                                                                       |
| [Chatbot_development_edition.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/Chatbot_development_edition.py) | The `Chatbot_development_edition.py` code snippet is a FastAPI endpoint that utilizes a ConversationRetrievalChain to implement a chatbot. It connects to a MongoDB Atlas instance, uses OpenAI for language modeling, and retrieves relevant documents based on user queries. The chatbot provides answers and prints metadata from the retrieved documents.                                                                                                                                                                                                                                      |
| [Chatbot_tutorial_edition.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/Chatbot_tutorial_edition.py)       | This code snippet implements a chatbot using the Streamlit framework. It connects to a MongoDB database, loads chat data from a CSV file, and uses OpenAI's GPT-3.5 Turbo model for conversational responses. Users can input queries, and the chatbot generates relevant answers based on the chat history.                                                                                                                                                                                                                                                                                       |
| [chat_fastapi_app.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/chat_fastapi_app.py)                       | This code snippet is a FastAPI application that provides an endpoint (/query) for processing user queries and returning relevant answers. It connects to a MongoDB Atlas database and uses various libraries for embeddings, vector search, and chat models. The code parses input parameters, constructs search conditions, sets up the retrieval chain, and processes the query. It also handles error cases and returns the generated answer and updated chat history. The code outlines several TODOs for bug fixing, code simplification, Dockerizing the app, and exploring hosting options. |

</details>

<details closed><summary>src.machine_learning.chat.docker</summary>

| File                                                                                                                                               | Summary                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---                                                                                                                                                | ---                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [docker-compose.yaml](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/docker/docker-compose.yaml)                       | The code snippet in `src/machine_learning/chat/docker/docker-compose.yaml` is responsible for building and running a Docker container for the chat FastAPI application. It specifies the build context, Dockerfile, port mapping, and environment variables. The container is connected to the `app_network` bridge network.                                                                                                 |
| [requirements.txt](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/docker/requirements.txt)                             | This code snippet, located at src/machine_learning/chat/docker/requirements.txt, lists the required packages and their versions for the Docker container used in the chat machine learning module. It ensures that the necessary dependencies are installed to enable chat functionality within the application.                                                                                                             |
| [chat_fastapi_conversational.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/docker/chat_fastapi_conversational.py) | This code snippet is a FastAPI application that serves as a chatbot endpoint. It handles queries and returns responses using a ConversationalRetrievalChain from the langchain library. The chatbot interacts with MongoDB Atlas for text search and retrieval and utilizes the OpenAI GPT-3.5-turbo-16k language model for generating responses. The code also includes CORS middleware for enabling cross-origin requests. |
| [chat_fastapi.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/docker/chat_fastapi.py)                               | This code snippet is a FastAPI application that serves as a chatbot endpoint for querying a MongoDB database. It handles incoming requests, parses the query parameters, connects to the database, performs a search using a retrieval question-answering model, and returns the result. The application also includes CORS middleware for handling cross-origin requests.                                                   |
| [chat_fastapi.dockerfile](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/docker/chat_fastapi.dockerfile)               | This code snippet is a Dockerfile that sets up a container for the chat FastAPI service. It installs the required packages and specifies the command to run the service.                                                                                                                                                                                                                                                     |
| [.env](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/chat/docker/.env)                                                     | This code snippet, located at src/machine_learning/chat/docker/.env, contains environment variables for accessing external services such as Atlas and OpenAI. It provides the necessary authentication credentials for secure communication with these services within the chat module of the R2G repository architecture.                                                                                                   |

</details>

<details closed><summary>src.machine_learning.BERTopic</summary>

| File                                                                                                                                    | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---                                                                                                                                     | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [requirements.txt](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/requirements.txt)                     | The code snippet in src/machine_learning/BERTopic/requirements.txt specifies the required dependencies for the BERTopic module. It includes packages such as argparse, bertopic, sklearn, pandas, regex, and tqdm. These dependencies are necessary for the BERTopic module to function properly within the parent repository's architecture.                                                                                                                                                               |
| [testing.ipynb](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/testing.ipynb)                           | This code snippet is part of a repository called R2G and specifically resides in the frontend/chatgpt-backend directory. It includes files related to a chatbot implementation using ChatGPT technology. The code snippet includes a Python file called Tumen_Chatbot_development_edition.py and a JavaScript file called chatGpt.controller.js. Its main role is to handle the backend logic and control the ChatGPT chatbot.                                                                              |
| [runBERTopic.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/runBERTopic.py)                         | Error generating text for src/machine_learning/BERTopic/runBERTopic.py: Client error '400 Bad Request' for url 'https://api.openai.com/v1/chat/completions'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400                                                                                                                                                                                                                                                        |
| [preprocess_testing_data.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/preprocess_testing_data.py) | Error generating text for src/machine_learning/BERTopic/preprocess_testing_data.py: Client error '400 Bad Request' for url 'https://api.openai.com/v1/chat/completions'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400                                                                                                                                                                                                                                            |
| [upload_model.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/upload_model.py)                       | This code snippet is responsible for uploading a BERTopic model to the HuggingFace Hub. It loads the model, pushes it to the Hub, and also outputs a dictionary containing topic information. The code is located in the `src/machine_learning/BERTopic/upload_model.py` file of the repository.                                                                                                                                                                                                            |
| [mapping.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/mapping.py)                                 | This code snippet in `src/machine_learning/BERTopic/mapping.py` connects to a MongoDB database, retrieves documents from a collection, and assigns a fixed cluster based on certain keywords. The assigned cluster is then stored in the database.                                                                                                                                                                                                                                                          |
| [apply_BERTopic_mongoDB.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/apply_BERTopic_mongoDB.py)   | The `apply_BERTopic_mongoDB.py` code snippet is responsible for fetching data from a MongoDB database, using BERTopic to predict topics for the messages, and updating the predicted topics back to the database in chunks. It utilizes multithreading for efficient processing.                                                                                                                                                                                                                            |
| [runsemi-supervised-bert.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/runsemi-supervised-bert.py) | The code snippet `runsemi-supervised-bert.py` is part of the BERTopic module in the repository. It defines a class that fits a BERTopic model, visualizes the results, and saves the model and representative documents to disk. The code uses UMAP for dimension reduction, HDBSCAN for clustering, and CountVectorizer for preprocessing. It also downloads NLTK resources and defines a list of stopwords. The results are saved as HTML files and an Excel spreadsheet, and the model is saved to disk. |
| [validation.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/BERTopic/validation.py)                           | The code snippet `chat_fastapi_app.py` in the `frontend/chatgpt-backend` directory serves as the backend implementation for the chat functionality of the R2G repository. It provides a FastAPI application that enables chatbot interactions with users.                                                                                                                                                                                                                                                   |

</details>

<details closed><summary>src.machine_learning.sentiment</summary>

| File                                                                                                                                                                                 | Summary                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                                                                                  | ---                                                                                                                                                                                                                                                                                                                                                                             |
| [sentiment.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/sentiment/sentiment.py)                                                                         | This code snippet is a Sentiment Classifier module that utilizes pre-trained transformer models to predict sentiment and emotion. It loads and processes data from an input file, performs sentiment and emotion inference, and saves the results. The module can be used in a larger architecture for sentiment analysis tasks.                                                |
| [cardiffnlp-twitter-xlm-roberta-base-sentiment.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/sentiment/cardiffnlp-twitter-xlm-roberta-base-sentiment.py) | This code snippet contains a class called `CardiffXLMRobertaModelBaseSentiment` that performs sentiment analysis using the CardiffNLP Twitter XLM-RoBERTa base model. It provides methods to classify sentiments for different data types such as Telegram, Twitter, and Google News. The sentiment analysis is done by utilizing the Transformers library's pipeline function. |
| [cardiffnlp-twitter-roberta-base-emotion.py](https://github.com/GIQSubmission/R2G/blob/master/src/machine_learning/sentiment/cardiffnlp-twitter-roberta-base-emotion.py)             | This code snippet is a class implementation for a sentiment classification model based on the CardiffNLP's Twitter RoBERTa base model. It provides methods to classify emotions in data from various sources (Telegram, Twitter, Google News). The code takes input data, runs it through the model, and outputs the results.                                                   |

</details>

<details closed><summary>src.helper.scraping.telegram_tools</summary>

| File                                                                                                                                                     | Summary                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                                                      | ---                                                                                                                                                                                                                                                                                                                                             |
| [requirements.txt](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/telegram_tools/requirements.txt)                                 | The `requirements.txt` file at `src/helper/scraping/telegram_tools` specifies the dependencies needed for scraping Telegram data, including packages for argument parsing, database interaction, and async programming.                                                                                                                         |
| [extractSateAndCity.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/telegram_tools/extractSateAndCity.py)                       | This code snippet is part of the repository's architecture for scraping and processing Telegram chats. It extracts the state and city information from the chats and stores it either in a local file or a MongoDB database. The code utilizes geosky library for mapping states and cities to their respective countries.                      |
| [scrapeTelegramChannelMessages.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/telegram_tools/scrapeTelegramChannelMessages.py) | This code snippet scrapes message texts and metadata from Telegram channels specified in an input file. It uses the Telethon library to interact with the Telegram API and saves the scraped data to a MongoDB database. Additionally, it extracts the state and city information from the chat names and performs text embedding using OpenAI. |
| [generateTelegramStringToken.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/telegram_tools/generateTelegramStringToken.py)     | The code snippet in the file `generateTelegramStringToken.py` generates a string token for accessing the Telegram API. It uses the `telethon` library and requires the user's phone number and login code.                                                                                                                                      |

</details>

<details closed><summary>src.helper.scraping.twitter_tools</summary>

| File                                                                                                                                | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                                 | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [count_tweets_single.sh](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/twitter_tools/count_tweets_single.sh) | The code snippet count_tweets_single.sh is located in the src/helper/scraping/twitter_tools directory. It counts the number of tweets that match a specific query and filters them based on start time and keywords. It uses the search_tweets.py script and requires a credential file. The result is printed as the sum of tweet counts.                                                                                                                      |
| [scrape_tweets.ipynb](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/twitter_tools/scrape_tweets.ipynb)       | The code snippet in the chatgpt-backend directory of the R2G repository is responsible for implementing the backend logic of a chatbot using GPT language generation. It includes files such as Tumen_Chatbot_development_edition.py, chatGpt.controller.js, chat_fastapi_app.py, and index.js. The code handles the communication between the frontend and the GPT model, enabling chatbot functionality.                                                      |
| [search_tweets.sh](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/twitter_tools/search_tweets.sh)             | The code snippet search_tweets.sh is located in the src/helper/scraping/twitter_tools directory of the parent repository. It is a bash script that utilizes a Python script to scrape tweets from Twitter based on provided queries. The tweets are stored in JSON files and can be converted to CSV format. The script also keeps track of the total number of tweets scraped.                                                                                 |
| [count_tweets.sh](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/twitter_tools/count_tweets.sh)               | This code snippet, located at src/helper/scraping/twitter_tools/count_tweets.sh, counts the number of tweets based on specified queries. It uses Twitter API to search for tweets and outputs the total count.                                                                                                                                                                                                                                                  |
| [search_tweets.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/twitter_tools/search_tweets.py)             | This code snippet is located in the `helper/scraping/twitter_tools` directory of the parent repository. It is a Python script that provides functionality for searching tweets using the Twitter API. The script takes various command-line arguments such as search queries, date ranges, and output formats to customize the search. It loads credentials, reads configuration files, and executes the search, returning the results in the specified format. |
| [create_tweets_csv.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/twitter_tools/create_tweets_csv.py)     | This code snippet, located at src/helper/scraping/twitter_tools/create_tweets_csv.py, processes a query file in JSON format and extracts relevant data from multiple JSON files. It then writes the extracted data into a CSV file and prints the number of tweets and unique tweets.                                                                                                                                                                           |

</details>

<details closed><summary>src.helper.scraping.traditional_news_tools.google</summary>

| File                                                                                                                                                                            | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---                                                                                                                                                                             | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [googlescapper.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/traditional_news_tools/google/googlescapper.py)                                         | This code snippet is located in the `src/helper/scraping/traditional_news_tools/google/googlescapper.py` file and is responsible for scraping news articles from Google using specific search terms for different countries and languages. The scraped data is stored in a MongoDB database. The code uses multithreading for efficient processing and handles error cases. The script also checks the size of the database and exits if it exceeds a predefined threshold. |
| [scrapeGoogleNewsPageLinksSoup.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/traditional_news_tools/google/scrapeGoogleNewsPageLinksSoup.py)         | This code snippet scrapes Google News articles and saves them as files. It utilizes concurrent threading for efficiency and handles connection errors. The goal is to gather data from Google News for further analysis.                                                                                                                                                                                                                                                    |
| [scrapeGoogleNews.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/traditional_news_tools/google/scrapeGoogleNews.py)                                   | The code snippet `scrapeGoogleNews.py` is part of the `R2G` repository's architecture. It scrapes news articles from Google News using parallel execution and saves the results to a CSV file. It supports multiple countries and languages, and handles connection errors and timeouts.                                                                                                                                                                                    |
| [preprocessMediaData.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/traditional_news_tools/google/preprocessMediaData.py)                             | This code snippet, located at `src/helper/scraping/traditional_news_tools/google/preprocessMediaData.py`, filters and processes news article data. It reads articles from a directory, checks for specific keywords, and writes the filtered articles to a new directory.                                                                                                                                                                                                   |
| [scrapeGoogleNewsPageLinksSelenium.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/traditional_news_tools/google/scrapeGoogleNewsPageLinksSelenium.py) | This code snippet is responsible for scraping and saving news article text from Google search results page using Selenium. It utilizes multithreading to process multiple URLs concurrently and writes the text to individual text files.                                                                                                                                                                                                                                   |

</details>

<details closed><summary>src.helper.scraping.traditional_news_tools.GDELT</summary>

| File                                                                                                                                   | Summary                                                                                                                                                                                                                                                         |
| ---                                                                                                                                    | ---                                                                                                                                                                                                                                                             |
| [scrapingGDELT.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/scraping/traditional_news_tools/GDELT/scrapingGDELT.py) | This code snippet in the file scrapingGDELT.py connects to Google BigQuery, executes a query to retrieve events data from GDELT, and prints the results. It is part of the src/helper/scraping/traditional_news_tools/GDELT module in the repository structure. |

</details>

<details closed><summary>src.helper.textcleaner</summary>

| File                                                                                             | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---                                                                                              | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [cleaner.py](https://github.com/GIQSubmission/R2G/blob/master/src/helper/textcleaner/cleaner.py) | This code snippet is part of the parent repository and is located at src/helper/textcleaner/cleaner.py. It contains a Cleaner class that has various methods for cleaning text data. The methods handle tasks such as removing URLs, correcting spelling mistakes, removing emojis, dropping sequences without words, and cleaning news articles from a given folder. The main() function parses command-line arguments and initializes an instance of the Cleaner class to run the data cleaning tasks based on the specified data type. |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version x.y.z`

###  Installation

1. Clone the R2G repository:

```sh
git clone https://github.com/GIQSubmission/R2G
```

2. Change to the project directory:

```sh
cd R2G
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

###  Running R2G

Use the following command to run R2G:

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

- **[Submit Pull Requests](https://github/GIQSubmission/R2G/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github/GIQSubmission/R2G/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github/GIQSubmission/R2G/issues)**: Submit bugs found or log feature requests for R2g.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/GIQSubmission/R2G
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
