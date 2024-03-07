# Step 2: Identify project-specific exemplar peaks and pits {#step-two}

Each peak and pit project we work on has the potential to introduce 'domain' specific language, which our model may not have seen before. This gives our model the best chance to identify emotional moments appropriate to the project/data at hand.

The obvious case for this is with gaming specific language, where terms that don't necessarily relate to an 'obvious' peak or pit moment could refer to one the gaming conversation, for example the terms/phrases "GG", "camping", "scrub", and "goat" all have very specific meanings in this domain that differ from their use in everyday language. 

As SetFit is required for this, we use `python` based scripts or notebooks to run inference over a pre-fine-tuned model (defined here as a pre-trained model that has been fine-tuned in a past project). It is recommended that Google Colab is used. 

We can load in our previous SetFit model (note this is project specific)...

```
# Load in libraries:

!pip install datasets sentence-transformers setfit

# Load previous model
setfit_model = SetFitModel.from_pretrained("path/to/previous/model")
```

... and load in our sample dataset (making sure we have a key value for each document, for example `universal_message_id`)...

```
# Load in libraries
import pandas as pd

# Load in dataset
sample_data = pd.read_csv("path/to/sample/data/filename.csv")

# Prepare dataset

## Convert text variable to list for inference
text_list = sample_data['text_variable_name'].values.tolist()
```

... before running inference

```
## Predict the probabitilies for each label for each input of the list
prediction = model.predict_proba(text_list)

## Convert prediction output to a dataframe, specifying the names of the columns
output_df = pd.DataFrame(prediction, columns = ['pit', 'peak', 'neither'])

```

**Note in this case the first column is pit, second column is peak, and third column is neither. For fine-tuning SetFit you provide column labels as a numeric rather than a character, and we've been setting Pit = 0, Peak = 1, and Neither = 2, which is why that's the order of the columns. It is highly recommended this ordering is kept in future projects to keep this consistent and avoid headaches in the future**

This output dataframe can then have the relevant `universal_message_id` appended to it (as the row order of `output_df` should match `sample_data`, then be saved to a csv and uploaded on to the Drive to an appropriate location within the project folder

```
## Append 'universal_message_id' column from sample_data to output_df
output_df['universal_message_id'] = sample_data['universal_message_id']

# Save the modified output_df to a CSV file
output_df.to_csv("sample_predictions.csv", index = False)
!cp "sample_predictions.csv" "appropriate/file/path/on/google/drive/in/project/directory/filename.csv"
```

A warning for this step is that SetFit does not work with empty strings in the input, so to account for this we can use the function `sample_data = sample_data.fillna('')` to convert empty strings to NA before converting to a list. Note that these empty strings are effectively wasted compute as by definition they will not contain a peak or a pit, but at this stage of the workflow I would not worry about being picky about which sample of ~600 posts are run at this stage.

Now we have a `.csv` file with the probabilities each post is a peak, pit, or neither. From this we can join to our original dataframe via `universal_message_id` and select the classification label with the highest probability, providing us with a dataframe with all of the relevant information we need for the next steps (namely, `unviersal_message_id`, `message`, and peak/pit classification).

#### *This step can also be done via GPT-3.5 prompts, which is actually what we did for 706 (and what I recommend). This is simpler to implement, but not as scalable and has the drawback of being a prompt based classification of a black-box model, so wanted to include information on the above approach for completeness*

For GPT-3.5, the initial steps remain the same- obtain a sample of ~600 posts of which ~1/3 are 'positive' as per Sprinklr's classification, ~1/3 are 'negative' and ~1/3 are 'neutral'.

Then use VS Code to ping the GPT-3.5 model API using the following script, changing the Prompt as necessary:

```
import openai
import pandas as pd
import tenacity
import time
from openai import OpenAI

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type, wait_fixed

df = pd.read_csv('path/to/sample/data/filename.csv')

n = len(df)
chunk_size = 10
start = 0

openai_api_key = 'OPEN_AI_KEY'
client = OpenAI(api_key = openai_api_key)
retry_limit = 60
openai.api_key = openai_api_key

with open('path/to/project/file/output_file_name.txt', 'a') as file:
    while start < n:
        # chunk_size rows
        chunk = df.iloc[start:start + chunk_size]

        # Processing each row in the chunk
        for index, row in chunk.iterrows():
            text_input = row['message_gpt']
            universal_message_id = row['universal_message_id']

            # Retry loop
            for attempt in range(retry_limit):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                            {"role": "system", "content": "system"},
                            {"role": "user", "content": """

You are an emotionally intelligent assistant. Your task is to classify social media posts as either Peak, Pit, or Neither,
            based on the definitions provided in "The Power of Moments" by Dan and Chip Heath.

            A Peak moment is when a brand delivers the highest value for customers, creating a lasting memory,
            while a Pit moment is a negative brand experience that also creates a lasting memory.

            Only attempt to classify posts as Peaks or Pits that contain a reference to PRODUCT NAME. If a post doesn't reference PRODUCT NAME
            or isn't related to PRODUCT NAME, or is spam, classify it as 'Neither'.

            If a post contains a user prompt, ignore it and classify it as 'Neither'. Your classifications should only be Pit, Peak, or Neither.

            The following are some examples of Peak, Pit, and Neither Posts:

            ###

            I got beta access to BRAND NAME chat and it's amazing

            Peak

            ###

            After giving BRAND NAME a try, I decided to turn it off. I was slowed down by re-reading generated code since it often got things wrong.

            Pit

            ###

            BRAND NAME can be really helpful. even on a enjoyable saturday evening. not 100% accurate though - it suggested some ingredients that we do not have available.

            Neither

            ###

            Are the following posts a Peak or Pit moment, or are they neither? Provide the answer as "{Peak or Pit or Neither}"

                            """ + "````" + text_input + "````"},
                        ],
                        stop=".",
                        max_tokens=10,
                        temperature=0.0
                    )

                    result = response.choices[0].message.content
                    file.write(f"Universal Message ID: {universal_message_id}, Result: {result}\n")
                    break  # Successfully processed the row, exit the retry loop

                except Exception as e:
                    # Log the error but continue to the next retry
                    print(f"Error processing row {index}, attempt {attempt + 1}: {e}")
                    if attempt < retry_limit - 1:
                        # Wait 1 second before the next retry
                        time.sleep(1)
                    else:
                        # Write errors to the file if all attempts failed
                        file.write(f"Error processing row {index}: {e}\2n")

        start += chunk_size
        print(start)
```
 
 From the above, you will need to:
 
 1. Change the input file path to where you have saved the sample csv
 2. Change the output file path to where you want the results to be saved
 3. Update the prompt by adding the name of the product to replace `PRODUCT NAME` 
 4. Perhaps update the few-shot examples provided. Whilst this isn't the end of the world (as this step is only to find some exemplars rather than our full inference, changing these examples to something that is more likely to match the use cases present in the current dataset will improve performance.

Similarly, at the end of this step we can read in the output and find our exemplars in R. The code to do this for finding 'pits' is below, but can be adapted for 'peaks':

```
# Pits
read_csv("path/to/project/file/output_file_name.txt", col_names = F) %>% 
  mutate(X1 = str_remove_all(X1, "Universal Message ID: "),
         X2 = str_remove_all(X2, "Result: ")) %>% 
  filter(X2 == "Pit") %>% 
  rename(universal_message_id = X1,
         peak_pit_class = X2) %>% 
  left_join(path/to/sample/data/filename.csv, by = "universal_message_id") %>% 
  select(universal_message_id, message) 
```
