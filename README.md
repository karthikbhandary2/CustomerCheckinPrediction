## CustomerCheckinPrediction
In this project I predict whether a customer checks into the hotel they booked or not using the features provided (The datasets are available in this repo). I specifically used NLP model to complete the task. I also deployed the model on to the web.

### Steps I followed:
- First imported the required packages
- Read the train and test data omitting the index column.
- Combined both train and test to make it easy to work.
- Took a look at how the data is and their dtypes by using the .info() method.
- I took a look at the unique values of all the columns to see if there is any need of data cleaning.
- I did some EDA(available in the colab notebook)
- I droped `ID`, `Unnamed: 0` and `Nationality` since they do not contribute to the model.
- I filled the missing values in the Age with the mean of that column.
- I then replaced the values which are greater than one in the column `BookingsCheckedIn` which is the target column. I did the same with `BookingsNoShowed` and `BookingsCanceled`
- I then encoded the `DistributionChannel` and `MarketSegment` columns using the `LabelEncoder()` from `sklearn.preprocessing`.
- I created a pickle file for each of the encoded column to be used in the `app.py`(can be found in model.py)
- I divided the dataset into `X` and `y`.
- I scaled `X` with the help of `MinMaxScaler` from `sklearn.preprocessing`
- I applied `train_test_split()` from `sklearn.model_selection`, where 80% is training data and remaining 20% is the testing data.
- I instantiated the `MultinomialNB()` which is a NLP model from `sklearn.naive_bayes`
- I fitted the training data.
- I then used the `.predict(X_test)`.
- I made every prediction that is greater than 0.5 to 1.
- I then made a `classification_report`.
- The model gave an accuracy of 1.0 and precision of 1.0 as well.
- Finally I pickled the model.

### In app.py (Flask was used to deploy the model)
- This file is used to deploy the model on to the web.
- In this the pickle files created above are first imported.
- I created a function `home()` to render the home page.
- `predict()` is used to make prediction by collecting the inputs from the website and then storing that in a list.
- We then do some transformations using the pickled encoder.
- Then pass all the features to make predictions and return the output.

### In index.html
- Contains all the script for the front end of the app.
- It also uses the css file.

All of this is used and hosted with the help of `heroku`. We basically linked this github repo and done!! Here is the link to the app: https://customer-check-in-prediction.herokuapp.com/

## Instructions in case you want to run this on your device:
- first you have to create a virtual environment you can do that by using the following command: `py -3 -m venv venv` (for windows)
- That will create a virtual environmment. You can activate it by using the command: `venv\Scripts\activate` (for windows)
- This will activate the virtual environment. Next run the file `model.py` by using: `python model.py`
- After running that run the `app.py` using: `python app.py` This will give you a link, using which you can visit the web app.

## Bonus
I think the difficult problem that I got around doing this is the deployment part. I was not ready for it at all. Like I said in the audio round. I once deployed my portfolio website using flask. I thought it would be easy this time as well as it was then but it was not. This doesn't even compare to it like I had completely no idea on how to do it. 
In particular, I had trouble with using the `scaler` and to use it to transform the features. I had to go throught YouTube, stackoverflow, discord to get it done. I had to get a lot of pointers from discord which really helped in me developing like they only gave me directions and I had to implement them on my own. I have to say this was not easy, but I am glad that I was able to complete this project since now I can say I am better than I was before(not much but it is honest work😉) When I finally got it I felt really happy.

**Back Propagation**: 
 It is nothing but the practice of fine tuning the weights of a neural network based on the error rate or loss that the neural networks gets from the previous epochs. If there are Nulls in the dataset we can counter it in many ways. Some of them are:
 - We can just drop them.
 - We can use random data may be use gaussian distribution.
 - We can fill them up with avgerage value.
