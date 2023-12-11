# Project-RecommendationSystem


## Project Overview

![Image](https://lp2m.uma.ac.id/wp-content/uploads/2022/04/ManfaatBuku.jpg)

In this era, reading is a fundamental skill that should be mastered by each individual. Individuals who read frequently tend to have critical, creative, and innovative thinking. Each individual has different literacy skills influenced by the environment in which they grow. Every country has different levels of interest in reading. Reading books can broaden insights, opening doors to civilization in society to achieve success. With various benefits derived from the habit of reading, it is essential for us to improve literacy by reading books each year.

The rapid development of digital technology has made books not only available physically but also accessible digitally anywhere and anytime. We can access books through applications or websites that provide free or paid books. Therefore, a website or application providing digital books requires a system that can recommend books based on user preferences. This recommendation system is not only essential for improving user satisfaction with the website or application but also for enhancing the reading habits of users, which can contribute to increasing the literacy rate in Indonesia, which is currently relatively low. According to a survey conducted by the Program for International Student Assessment (PISA) released by the Organization for Economic Co-operation and Development (OECD) in 2019, Indonesia ranked 62nd out of 70 countries. This implies that Indonesia is among the 10 countries with low literacy rates. Therefore, in this recommendation system project, the author will create a book title recommendation system using Content-Based Filtering and Collaborative Filtering models, aiming to benefit readers and facilitate them in finding books based on their preferences.

## Business Understanding
---
### Problem Statements

Based on the background outlined above, the problem statements to be addressed in this project include:

* How to process data effectively to be used in building a good recommendation system model?
* How to create a machine learning system that can provide a set of book recommendations based on the names of authors the user has read?
* How to create a machine learning system that can recommend book titles according to user preferences based on existing ratings?

### Goals

The goals of the problem statements are as follows:

- Process data effectively to be used in building a good book recommendation system model.
- Build a machine learning model with a book recommendation system based on the names of authors the user has read.
- Build a machine learning model with a book recommendation system for book titles that match user preferences based on existing ratings.

### Solution Approach

The solutions that can be applied to achieve the above goals are as follows:

* Analyze the data (data preparation) to understand the available data, such as checking for missing values and duplicate data.
* Process the data, such as normalizing rating data, to make it easily processed by the model.
* Build a recommendation system using two commonly used techniques: Content-Based Filtering and Collaborative Filtering:

1. **Content-Based Filtering**
   * The idea of this approach is to recommend items similar to items that the user has liked in the past. Developing a model using this approach is done to generate book recommendations based on the names of authors the user has read.

2. **Collaborative Filtering**
   * This approach is based on user community ratings. Additionally, it does not require attributes for each item, as in Content-Based Filtering systems. The goal of this approach is to generate a set of book recommendations that match user preferences based on previously given ratings.

## Data Understanding
---
Based on the dataset source: [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), the following information is obtained:

Table 1. Dataset Information
| Type | Description |
| -------- | -------- |
| Dataset Source | Book Recommendation Dataset: [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) [3] |
| Owner/Collaborator | Möbius |
| Usability | 10.0 |
| Dataset Origin | [Book-Crossing](https://www.bookcrossing.com/?) |
| License | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| File Type and Size | .zip (25 MB) |
| Number of Dataset Files | 3 Files (CSV) |

The three dataset files are as follows:

1. Books.csv
2. Ratings.csv
3. Users.csv

In this project, the author will only use two dataset files:

**1. Books**
The variable descriptions in the **Books.csv** file are as follows:

* ISBN: Unique book number or International Standard Book Number
* Book-Title: Book title
* Book-Author: Book author
* Year-Of-Publication: Year the book was published
* Publisher: Book publisher
* Image-URL-S: URL of the small book cover image
* Image-URL-M: URL of the medium-sized book cover image
* Image-URL-L: URL of the large-sized book cover image



**2. Ratings**
The variable descriptions in the **Ratings.csv** file are as follows:

* User-ID: Unique user identifier
* ISBN: Unique book number or International Standard Book Number
* Book-Rating: User's rating for the book (scale: 1-10)

The **Users.csv** file contains information about user demographics and is not used in this project. The analysis will focus on the Books and Ratings datasets.

## Data Preparation
---
### Data Cleaning
#### Handling Missing Values

The first step in data preparation is to check for missing values in the dataset and decide how to handle them. Missing values can arise from various reasons, such as data entry errors, incomplete data, or intentional omissions.

The dataset is checked for missing values, and appropriate actions are taken based on the extent and nature of missingness.

```python
# Check for missing values in the Books dataset
books.isnull().sum()
```

#### Handling Duplicate Data

Duplicate data refers to identical rows or entries in the dataset. Duplicate data can arise from data collection methods or errors in data entry. It is crucial to identify and handle duplicate data appropriately to prevent biases and inaccuracies in the analysis.

```python
# Check for duplicate rows in the Books dataset
books.duplicated().sum()
```

### Exploratory Data Analysis (EDA)
#### Overview of Books Dataset

Before building the recommendation system, it is essential to gain insights into the distribution and characteristics of the data. Exploratory Data Analysis (EDA) involves visualizing and summarizing key aspects of the dataset.

```python
# Overview of the Books dataset
books.head()
```

#### Distribution of Ratings

Understanding the distribution of book ratings in the dataset is crucial for designing the recommendation system. A balanced distribution ensures that the model can provide meaningful recommendations for a wide range of user preferences.

```python
# Distribution of book ratings
ratings['Book-Rating'].value_counts().sort_index().plot(kind='bar')
```

### Data Preprocessing for Recommendation Systems
#### Content-Based Filtering

Content-Based Filtering involves recommending items similar to those the user has liked in the past. In the context of books, this can be achieved by considering features such as book titles, authors, and genres.

```python
# Create a TF-IDF matrix for book titles
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books['Book-Title'])
```

#### Collaborative Filtering

Collaborative Filtering is based on user community ratings. It recommends items based on the preferences and behavior of similar users. In this project, collaborative filtering is implemented using the Surprise library.

```python
# Create a Surprise dataset
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
```

## Model Development
---
### Content-Based Filtering Model

The Content-Based Filtering model is developed using the TF-IDF matrix for book titles.

```python
# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

### Collaborative Filtering Model

The Collaborative Filtering model is developed using the Surprise library and the SVD algorithm.

```python
# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Build the collaborative filtering model using SVD
model = SVD()
model.fit(trainset)
```

## Model Evaluation
---
### Content-Based Filtering Evaluation

The Content-Based Filtering model is evaluated based on the cosine similarity between book titles.

```python
# Calculate cosine similarity for a specific book
book_index = books[books['Book-Title'] == 'Selected Poems'].index[0]
cosine_scores = list(enumerate(cosine_sim[book_index]))
```

### Collaborative Filtering Evaluation

The Collaborative Filtering model is evaluated using the Root Mean Squared Error (RMSE) on the test set.

```python
# Evaluate the collaborative filtering model using RMSE
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
```

## Conclusion
---
In conclusion, the recommendation system project aims to provide users with personalized book recommendations based on their preferences. Two approaches, Content-Based Filtering and Collaborative Filtering, are implemented to achieve this goal. The data is prepared by handling missing values and duplicate data, and exploratory data analysis is performed to gain insights into the distribution of book ratings.

The Content-Based Filtering model uses TF-IDF vectors for book titles to calculate cosine similarity, while the Collaborative Filtering model employs the Surprise library with the SVD algorithm. The models are evaluated using relevant metrics, such as cosine similarity and RMSE.

The successful implementation of these recommendation systems can enhance the user experience on a book-related platform, leading to increased user satisfaction and engagement. Future improvements may involve fine-tuning the models, incorporating additional features, and exploring hybrid recommendation systems for more robust and accurate results. Overall, the project contributes to the promotion of literacy by encouraging users to discover and enjoy books tailored to their interests.

## References
---
[1] Möbius. (n.d.). Book Recommendation Dataset. Kaggle. [https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

[2] BookCrossing. (n.d.). [https://www.bookcrossing.com/](https://www.bookcrossing.com/)

[3] Creative Commons. (n.d.). CC0: Public Domain. [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)

[4] Organization for Economic Co-operation and Development (OECD). (2019). PISA 2018 Results (Volume I): What Students Know and Can Do. [https://www.oecd.org/pisa/publications/pisa-2018-results-volume-i-what-students-know-and-can-do-d48b5f07-en.htm](https://www.oecd.org/pisa/publications/pisa-2018-results-volume-i-what-students-know-and-can-do-d48b5f07-en.htm)
