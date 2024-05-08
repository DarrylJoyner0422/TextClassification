
<h1>Text Classification to Detect Sarcasm</h1>

 ### [YouTube Demonstration](https://youtu.be/7eJexJVCqJo)

<h2>Description</h2>

In this project, I will train and analyze a text classifier to predict sarcasm in Reddit comments using the SARC 2.0 dataset. The classifier will determine if a comment is sarcastic based on explicit sarcasm markers in the message. Sarcasm detection is challenging because it is highly subjective, often misunderstood, and depends on speaker intent and listener interpretation. Despite these complexities, this project will simplify the task by focusing solely on explicit sarcasm markers without considering the surrounding discussion context.
<br />


<h2>Python Libraries Used</h2>

- <b>Matplotlib (matplotlib.pyplot): For plotting and data visualization.</b> 
- <b>Pandas (pandas): For data manipulation and analysis.</b>
- <b>Seaborn (seaborn): For statistical data visualization</b>
- <b>Scikit-learn (sklearn):</b>
- <b>TfidfVectorizer: For converting text data into TF-IDF feature vectors.</b>
- <b>ENGLISH_STOP_WORDS: A set of common English stop words.</b>
- <b>LogisticRegression: For building a logistic regression classifier.</b>
- <b>f1_score: For evaluating the classifier using the F1 score.</b>
- <b>make_pipeline: For creating a machine learning pipeline.</b>
- <b>tqdm: For displaying progress bars.</b>



<h2>Environments Used </h2>

- <b>Jupyter</b> (21H2)



<h2>Program walk-through:</h2>

<p align="left">
Load Python Libraries: <br/>
<img src="https://i.imgur.com/JTLR7y6.jpeg" height="50%" width="50%" alt="DJ Test "/>
<br />
<br />
Load the Data:  <br/>
<img src="https://i.imgur.com/JrBOY7a.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Examples of Sarcastic vs Not Sarcastic: <br/>
<img src="https://i.imgur.com/6YuMVlc.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Proof of Balance: The training data in train_df, along with the test set in bal_df, is balanced, in that there are an equal number of sarcastic and non-sarcastric labels. The test set in imb_df is imbalanced  <br/>
<img src="https://i.imgur.com/EXdW75H.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Store labels in lists for later input into classifier:  <br/>
<img src="https://i.imgur.com/uTwXL21.jpeg" height="40%" width="40%" alt="Disk Sanitization Steps"/>
<br />
<br />
Train a classifier:
Train a logistic regression classifier, using tf-idf reweighted features. 
I will consider two slightly-different modeling choices: one where I filter stopwords, and one where I do not. 
Get tf-idf features
First get the text features.

** Create a TfidfVectorizer object that only considers words occuring at least 100 times in the data, and that filters out scikit-learn's list of English stop words. I will name this "vectorizer_nostop".

Fit the vectorizer on, and use it to featurize the training data (train_df.text). I will call the output X_train_nostop.
Use the vectorizer to featurize the balanced and imbalanced test data. I will call the outputs X_bal_nostop and X_imb_nostop, respectively.
<br/>
<img src="https://i.imgur.com/VZPBHFk.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />


Train a Classifier:
Now I will train a logistic regression classifier.

Steps
Initialize a LogisticRegression object, called clf_nostop, and fit it on the training data
Generate predictions on the balanced and imbalanced test sets, storing the predictions as variables pred_bal_nostop and pred_imb_nostop, respectively.
Compute f1 scores for each set of prediction, and call the resultant scores f1_bal_nostop and f1_imb_nostop, respectively.

<br/>
<img src="https://i.imgur.com/4EouW40.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

The performance looks much worse on the imbalanced data. I will closely consider why this might be the case. 


<br />
<br />



Get tf-idf features, with stopwords:  
I will consider one alternate choice: instead of removing stopwords when featurizing the text I will leave them in. 
Reddit comments in the data are pretty short to start out with, so removing stopwords has a small impact on them. 
My goal is to see if keeping the stopwords in will result in better or worse model performance?

Steps:
Create a TfidfVectorizer object that only considers words occuring at least 100 times in the data, but that does not filter for stopwords. Call it vectorizer_full.
Fit the vectorizer on, and use it to featurize the training data . Call the output X_train_full.
Use the vectorizer to featurize the balanced and imbalanced test data. Call the outputs X_bal_full and X_imb_full, respectively.

<br/>
<img src="https://i.imgur.com/0juJMjK.jpeg" height="80%" width="80%" alt="1.3"/>
<img src="https://i.imgur.com/9E0xfw4.jpeg" height="80%" width="80%" alt="1.3b"/>

We get slightly more features than before -- which makes sense, given that we do not remove stopwords.



<br />
<br />
Train and evaluate classifier, with stopwords included:  <br/>
Now I'll train a logistic regression classifier on the new set of features.

Steps: <br/>
Initialize a LogisticRegression object, called clf_full, and fit it on the training data with stopwords
Generate predictions on the balanced and imbalanced test sets, storing the predictions as variables pred_bal_full and pred_imb_full, respectively.
Compute f1 scores for each set of prediction, and call the resultant scores f1_bal_full and f1_imb_full, respectively.

<br/>
<img src="https://i.imgur.com/SKMfIsR.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>





<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>


Findings = Note that relative to the no-stopwords case, we get slightly better model performance.





<br />
<br />
Examine Model Weights:  <br/>
I will examine what features are informative for the classifier. Since the vectorizer+classifier pipeline with stopwords performed slightly better, I will focus on it for the rest of the development.

Assign variables and values:

** coefs: an array consisting of the feature weights learned by the clf_full classifier.
** features: an array consisting of the feature names, i.e., words, corresponding to each entry of coefs

<img src="https://i.imgur.com/BzXvagP.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>



I will store the features and model weights in the following dataframe:

<img src="https://i.imgur.com/6TVy6BD.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>





<br />
<br />
Examine Most Informative Features:  <br/>
I will define two subsets of coef_df:

** pos_coef_df: the 25 features that are the most informative that a text is sarcastic, ordered from more to less informative;
** neg_coef_df: the 25 features that are most informative that a text is not sarcastic, ordered from more to less informative

<img src="https://i.imgur.com/KAikGQe.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

Code:<br />
<img src="https://i.imgur.com/GOONktQ.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<img src="https://i.imgur.com/dg24YzX.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<img src="https://i.imgur.com/KR3pSBH.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>





2.3-----------------------------








<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>


<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>


<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>


<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<br />
<br />
NEXT PICTURE:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>







</p>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
