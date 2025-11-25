import pandas as pd
import tiktoken
import openai
import os
from dotenv import load_dotenv 

max_token_count = 4097
max_response_length = 150

COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#Load CSV data
df = pd.read_csv('data/2023_fashion_trends.csv')

#Extract text data columnn
df_cleaned = df["Trends"]

print(df.head())


def create_prompt(question, df, max_token_count):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model
    """
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Count the number of tokens in the prompt template and question
    prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context: 

{}

---

Question: {}
Answer:"""

    current_token_count = len(tokenizer.encode(prompt_template)) + \
                            len(tokenizer.encode(question))

    context = []
    #for text in get_rows_sorted_by_relevance(question, df)["text"].values:
    for text in df:

        # Increase the counter based on the number of tokens in this row
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count

        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return current_token_count, prompt_template.format("\n\n###\n\n".join(context), question)

def answer_question(
    question, df, max_prompt_tokens=1800, max_answer_tokens=150
):
    """
    Given a question, a dataframe containing rows of text, and a maximum
    number of desired tokens in the prompt and response, return the
    answer to the question according to an OpenAI Completion model

    If the model produces an error, return an empty string
    """

    [num_tokens, prompt] = create_prompt(question, df, max_prompt_tokens - max_answer_tokens)

    try:
        response = openai.Completion.create(
            model=COMPLETION_MODEL_NAME,
            prompt=prompt,
            max_tokens=max_answer_tokens
        )
        return num_tokens, response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

#ChatGPT without provided fasion 2023 context:
question1 = "It's 2023, is a Canadian tuxedo in or out?"

#Obtain original result without context
response = openai.Completion.create(
    model=COMPLETION_MODEL_NAME,
    prompt=question1,
    max_tokens=max_response_length
)

#Add context from dataset
_ , prompt_with_context= create_prompt(question1, df_cleaned, max_token_count- max_response_length)

#Obtain result with context
response_with_context = openai.Completion.create(
    model=COMPLETION_MODEL_NAME,
    prompt=prompt_with_context,
    max_tokens=max_response_length
)

print("Question:\n" + question1 + "\n")
print("Answer Without Context:\n" + response["choices"][0]["text"].strip() + "\n")
print("Answer With Fashion Trends Context:\n" + response_with_context["choices"][0]["text"].strip())
