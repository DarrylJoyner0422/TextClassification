
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
Train and evaluate classifier, with stopwords included:  <br/>
Now I'll train a logistic regression classifier on the new set of features.

Steps: <br/>
Initialize a LogisticRegression object, called clf_full, and fit it on the training data with stopwords
Generate predictions on the balanced and imbalanced test sets, storing the predictions as variables pred_bal_full and pred_imb_full, respectively.
Compute f1 scores for each set of prediction, and call the resultant scores f1_bal_full and f1_imb_full, respectively.

<br/>
<img src="https://i.imgur.com/SKMfIsR.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>





<br />


Findings = Note that relative to the no-stopwords case, we get slightly better model performance.



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
<br/>
Examine Most Informative Features:  <br/>
I will define two subsets of coef_df: <br/>


** pos_coef_df: the 25 features that are the most informative that a text is sarcastic, ordered from more to less informative;<br/>
** neg_coef_df: the 25 features that are most informative that a text is not sarcastic, ordered from more to less informative

<img src="https://i.imgur.com/KAikGQe.jpeg" height="40%" width="40%" alt="Disk Sanitization Steps"/>

Code:<br />
<img src="https://i.imgur.com/GOONktQ.jpeg" height="%" width="80%" alt="Disk Sanitization Steps"/>

<img src="https://i.imgur.com/dg24YzX.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<img src="https://i.imgur.com/KR3pSBH.jpeg" height="60%" width="60%" alt="Disk Sanitization Steps"/>





<br />
Examine Most Informative Stopword Features:  <br/>
I will examine why the with-stopword approach performed better than the approach where stopwords were filtered out.
<br />

Steps:<br />
Come up with the subset of coef_df consisting of stopword features (per scikit-learn), and where the corresponding weights are especially high or low. 
Take especially high or low to mean above 1.5 or below -1.5. 
Call this subset stopword_feat_df and order it from most negative to most positive model weights.

<img src="https://i.imgur.com/qGcnK6z.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>




It seems like there are some stopwords that are very informative of (non)sarcasm.

In fact, in many settings, stopwords can encode a fair amount of linguistic or social information. So it's worth testing out the impacts of filtering stopwords out versus leaving them in.



<br />
Analyze classifier performance:  <br/>
I will examine the predictions that the model outputs, and what sorts of errors it makes. 
I would like to explore the poor performance on the imbalanced test set.

Steps:
Output the proportion of sarcasm labels and predictions
Compute the proportion of items in the imbalanced test set which are labeled sarcastic.
Comput  the proportion of items in that set which the model guessed was sarcastic. 
Call these values pr_label_is_sarcasm and pr_pred_is_sarcasm respectively.
<img src="https://i.imgur.com/KnUc9lT.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

The model predicts sarcasm much more often than sarcastic items actually exist in the data!

This test data is heavily imbalanced. The data I trained the model on was balanced, with close to 50% of items having a sarcastic label. Some of my prediction errors may have incurred due to this shift in label distribution between training and testing: the model was over-estimating how many items are sarcastic because that's what it expects from the distribution in the training data. In other words, it's miscalibrated.



Next I will consider a way of looking at this label distribution problem. 
In addition to outputting a predicted label, a logistic regression classifier also outputs a estimated probability that an item is of that label. In the below dataframe, I will collect the actual labels and predictions, along with the model's estimated probability that each item in the imbalanced set is sarcastic:<br />

<img src="https://i.imgur.com/MwjhMCa.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>



Since there are only two labels (sarcastic or not), the way the model converts the probability estimate to a prediction is, simply, output sarcastic if the probability is greater than 0.5.

If the model outputs a "sarcastic" prediction too often, then what would happen if I increased the cutoff beyond 0.5. 
In this event the model would only predict "sarcastic" for a fewer number of comments where it is especially confident, perhaps resulting in more accurate output. 
It is not good practice to tweak a model on the test data. But, for the purposes of this assignment I will experiment with lowering or raising the cutoff to build intuition.



<br />
<br />
Vary the Probability Estimate Cutoff:  <br/>

Steps: <br/>


get_cutoff_f1 will compute the f1 score, if the model predicts "sarcastic" only when its probability estimate is greater than or equal to a given cutoff.<br/>
get_pr_pos_preds will compute the proportion of items the models predicts is "sarcastic", if it makes that prediction when its probability estimate is greater than or equal to a given cutoff. <br/>
For both functions, the true and pr parameters are arrays containing the label of each item and the probability estimate for each item.

<img src="https://i.imgur.com/FKDdm76.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>


By default, the model's cutoff is 0.5, so the outputs of the functions when cutoff=0.5 should be equivalent to how well the model is currently doing. If I raise the cutoff slightly, here's what happens:<br />


<img src="https://i.imgur.com/Ft3dxAx.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>


I will graph what happens at different cutoff values: how do the proportion of predictions and the f1 score vary?<br />

<img src="https://i.imgur.com/78hgGtQ.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<img src="https://i.imgur.com/RzzI4qv.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

This shows that the proportion of items predicted as sarcastic decreases from 1 to 0 as the cutoff increases, following a sigmoid-like curve. The f1 score peaks somewhere in the middle.





<br />
<br />
Locate The Max f1 score:  <br/>

Computing the F1 score is essential here as it provides a robust measure of the classifier's accuracy. This is specifically in the context of the sarcasm detection where class distinctions can be subtle, ambiguous and subjective. The F1 score will allow us to combine precision and recall and will offer a single metric that balances the trade-off between the two. 

This balance is crucial in this dataset because sarcasm may not be consistently marked and the consequences of misclassifications—either missing sarcastic comments or falsely labeling genuine comments as sarcastic—can significantly impact the model's utility. By focusing on the F1 score, I will ensure that the classifier not only identifies sarcasm effectively but also minimizes errors in a way that is practical for real-world application. This is the vital step in fine-tuning and validating the model’s performance.<br />


Steps: <br/>
Assign to the following variables:

max_f1: the maximum f1 score, across all of the cutoffs sampled <br/>
max_at_cutoff: the value of the cutoff that achieves the maximum f1 score <br/>
pr_at_max: the proportion of items predicted as sarcastic, when the maximum f1 score is achieved.<br/>

<img src="https://i.imgur.com/6kocYEZ.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>


Results:
I show that for a certain value of cutoff (that's higher than 0.5), I get a better-looking f1 score, and fewer "sarcastic" predictions.

These results indicate that having a training dataset that better reflects the distribution of labels I expect at test time may result in a better-calibrated model. Even with this toggling, the f1 score only goes up so far -- suggesting that there may be additional signal the model is failing to capture, beyond the label distribution problem, and/or that the task is inherently ambiguous or difficult.



<br />
Examine Specific Model Errors:  <br/>

Now I will examine particular items in the imbalanced test data where the model made an error. To build intuition, I will focus on cases where the model was particularly confident in making that incorrect prediction.

I will store the comment texts in pred_df 


<img src="https://i.imgur.com/0xBSbWs.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>



There are two types of errors the model can make: <br />
1. false positives (where the label is "not sarcastic" but the model outputs "sarcastic")
2. false negatives (where the label is "sarcastic" but the model outputs "not sarcastic").

Get subsets of pred_df:

false_pos: the 10 items in the imbalanced data where the model makes a false positive error, ordered from more to less confident.
false_neg: the 10 items where the model makes a false negative error, ordered from more to less confident.
(higher or lower probabilities, as stored in pred_df, correspond to more confident predictions of "sarcastic" and of "not sarcastic")


<img src="https://i.imgur.com/LOmeioi.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<img src="https://i.imgur.com/8QS9jJm.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>


<img src="https://i.imgur.com/rASp7q9.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>



<img src="https://i.imgur.com/K98ZdLT.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>




I will use these to inspect the most confident model errors:


<img src="https://i.imgur.com/G0qEKsa.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<img src="https://i.imgur.com/K98ZdLT.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>


Reading through these examples I examine what accounts for these errors: the ability to model text, or some property of the data, or something about the nature of sarcasm in Reddit comments?

Questions: do people on Reddit consistently use /s when they are being sarcastic? If they do this some but not all of the time, but not all of the time, then

1. A model trained on this label might still learn a rough sense of what sarcasm looks like in text;
2. The model might make wrong predictions when the comment-writer has misapplied the label.

Another case to consider is if people only use /s for certain types of sarcasm. Suppose that /s is only applied to sarcastic comments when people are making jokes; otherwise, sarcastic comments are left unadorned. In that case, would a model trained on this label learn about signifiers of sarcasm, or signifiers of sarcastic comments written in a joke-y context?




<br />
<br />
Predict on Individual Items:  <br/>

One other way of checking that I understood what the model was doing is to see if I could think of inputs that result in different model predictions. The following code took a string text and output the model's prediction on it:



<img src="https://i.imgur.com/rO26Ze0.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>



In the below cell, I tested with two strings:

1. outputs_sarcastic will be predicted to be sarcastic by our model;
2. outputs_not_sarcastic will be predicted to be not sarcastic by our model.

Neither of these strings occurred anywhere in the training or test datasets 


<img src="https://i.imgur.com/RMVP3wy.jpeg" height="80%" width="80%" alt="Disk Sanitization Steps"/>








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
