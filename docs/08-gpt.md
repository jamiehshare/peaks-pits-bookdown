# The metal detector, GPT-3.5 (Step 6) {#step-six}

During [step 5](#setfit-inference) we obtained peak and pit classification using few-shot classification with SetFit. The benefit of this approach (as outlined previously) is its speed and ability to classify with very few labelled samples due to contrastive learning. 

However, during our iterations of peak and pit projects, we've realised that this step still classifies a fair amount of non-peak and pit posts incorrectly. This can cause noise in the downstream analyses and be very time consuming for us to further trudge through verbatims.

As such, the aim here is to further our confidence in our final list of peaks and pits to be *actually* peaks and pits. Remember before we explained that for SetFit, we focussed on **recall** being the most important measure in our business case? This is where we assume that GPT-3.5 enables us to remove the false positives due to it's incredibly high performance.

> Note: Using GPT-3.5 for inference, even over relatively few posts as in peaks and pits, is expensive both in terms of time and money. Preliminary tests have suggested it is in the order of magnitude of thousands of times slower than SetFit. It is for these reasons why we do not use GPT-x models from the get go, despite it's obvious incredible understanding of natural language.

Whilst prompt-based classification such as those with GPT-3.5 certainly has its drawbacks (dependency on prompt quality, prompt injections in posts, handling and version control of complex prompts, unexpected updates to the model weights rendering prompts ineffective), the benefits include increased flexibility in what we can ask the model to do. As such, in the absence of an accurate, cheap, and quick model to perform span detection, we have found that often posts identified as peaks/pits did indeed use peak/pit language, but the context of the moment was not related to the brand/product at the core of the research project. 

For example, take the post that we identified in the project 706, looking for peaks and pits relating to PowerPoint:

>This brings me so much happiness! Being a non-binary graduate student in STEM academia can be challenging at times. Despite using my they/them pronouns during introductions, emails, powerpoint presentations, name tags, etc. my identity is continuously mistaken. Community is key!

This is clearly a 'peak', however it is not accurate or valid to attribute this memorable moment to PowerPoint. Indeed, PowerPoint is merely mentioned in the post, but is not a core driver of the Peak which relates to feeling connection and being part of a community. This is as much a PowerPoint Peak as it is a Peak for the use of emails.

Therefore, we can engineer our prompt to include a caveat to say that the specific peak or pit moment must relate directly to the brand/product usage (if relevant).

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

From the above, as before, you will need to:

1. Change the input file path to where you have saved the sample csv
2. Change the output file path to where you want the results to be saved
3. Update the prompt by adding the name of the product to replace PRODUCT NAME
4. Update the few-shot examples provided to be suitable for the specific product

Then we can read in the output from GPT-3.5 as a dataframe, clean the dataframe, and join with the original dataframe too. 

```
df <- read_csv("path/to/project/file/output_file_name.txt", col_names = F)

# Renaming and transforming the dataframe
df_cleaned <- df %>%
  rename(
    universal_message_id = X1,
    classification = X2
  ) %>%
  mutate(
    universal_message_id = str_replace_all(universal_message_id, "Universal Message ID: ", ""),
    classification = str_replace_all(classification, "Result: ", ""),
    classification = case_when(
      str_detect(classification, "Pit") ~ "Pit",
      str_detect(classification, "Peak") ~ "Peak",
      TRUE ~ "Neither"
    )
  ) %>%
  distinct(universal_message_id, .keep_all = TRUE)

# Counting the number of each classification
df_classification_counts <- df_cleaned %>% 
  count(classification)

# Joining with original dataframe and rename columns (og_df is a dataframe with all posts with the key analytical columns such as universal_message_id, message, created_time etc)
df_output <- df_cleaned %>%
  left_join(og_df, by = "universal_message_id") %>%
  rename(
    gpt_classification = classification,
    setfit_classification = model_classification
  )
```
