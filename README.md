# RottenTomatoesReviewClassification

## Objective

The goal of this project is to learn how natural language text is classified using sentiment analysis. The project will aim to predict the classification of a review based on the words the critic used in their review, using reviews by accredited critics obtained from Rotten Tomatoes. In addition, rather than only using n-grams for the features, another feature will be created using sentiment information to see if performance of the model improves.

## Corpus

The corpus obtained from Kaggle consists of a .csv file, which contains 480,000 critic reviews from Rotten Tomatoes. There are two columns in the .csv file: the freshness (categorization of the review i.e. fresh) and the text of the review. The text of the review is a snippet of the actual review, which is displayed on the Rotten Tomatoes site with a link to the actual review. The data only consists of what is displayed on the Rotten Tomatoes site; thus, the review can be as short as a single sentence or as long as a short paragraph.

## Project Pipeline
1. Obtain Corpus
2. Preprocess Corpus
3. Feature Engineering
4. Model Building
5. Model Evaluation

## Results
Final results are documented in the project paper which is included in the repository. 
