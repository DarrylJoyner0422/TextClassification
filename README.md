
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
Sanitization complete:  <br/>
<img src="https://i.imgur.com/K71yaM2.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Observe the wiped disk:  <br/>
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
