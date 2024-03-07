# Step 5: Run inference over all project data {#step-five}

It is finally time to infer whether the project data contain peaks or pits by using our fine-tuned SetFit model to classify the posts.

Before doing this again we need to make sure we do some data cleaning on the project specific data. 

Broadly, this needs to match the high-level cleaning we did during fine-tuning stage:

* Mask brand/product mentions (using RoBERTa-based model [or similar] and `Rivendell` functions) 
* Remove hashtags #️⃣
* Remove mentions 💬
* Remove URLs 🌐
* Remove emojis 🐙
 
> Note: Currently all peak and pit projects have been done on Twitter or Reddit data, but if a project includes web/forum data quirky special characters, numbered usernames, structured quotes etc should also be removed.

Now we save this dataframe somewhere appropriate.

Okay now we can *finally* run inference. Note this code follows the same structure as the SetFit code in [step 2](#killer-examples):

```
import pandas as pd

# Load in dataset
input_df = pd.read_csv("path/to/sample/data/filename.csv")

input_df = input_df.fillna('')

# Load current model
SetFit_model = SetFitModel.from_pretrained("path/to/current/model")

## Convert text variable to list for inference
text_list = input_df['text_variable_name'].values.tolist()

## Predict the probabitilies for each label for each input of the list
prediction = SetFit_model.predict_proba(text_list)

## Convert prediction output to a dataframe, specifying the names of the columns
output_df = pd.DataFrame(prediction, columns = ['pit', 'peak', 'neither'])

## Append 'universal_message_id' column from sample_data to output_df
output_df['universal_message_id'] = input_df['universal_message_id']

# Save the modified output_df to a CSV file
output_df.to_csv("data_predictions.csv", index = False)
!cp "data_predictions.csv" "appropriate/file/path/on/google/drive/in/project/directory/filename.csv"
```

Now we have a .csv file with the probabilities each post is a peak, pit, or neither. From this we can join to our original dataframe via universal_message_id and select the classification label with the highest probability, providing us with a dataframe with all of the relevant information we need for the next steps (unviersal_message_id, message column, and peak/pit classification etc).
