import io
import os

import PyPDF2
from tqdm import tqdm


# Open the PDF file in read-binary mode
def convert_pdf_to_txt(step: int):
    if step == 1:
        # Check if the file already exists
        if not os.path.exists("document.txt"):
            # Write the text to a text file
            with open("document.pdf", "rb") as file:
                # Create a PDF object
                pdf = PyPDF2.PdfReader(file)
                # Iterate over every page
                for page in range(len(pdf.pages)):
                    # Extract the text from the page
                    text = pdf.pages[page].extract_text()
                    # Write the text to a text file
                    with open("document.txt", "a") as f:
                        f.write(text)

        # Open the text file
        with open("document.txt", "r") as file:
            text = file.read()
    if step == 2:
        # Open the text file
        with open("output.txt", "r") as file:
            text = file.read()

        with open("output_copy.txt", "w") as out_file:
            # Write the copy of the output file
            out_file.write(text)

    chunks = split_text_to_chunks(text=text, max_tokens=400)
    print("splitted chunks:", len(chunks))
    print("estimated cost: ~", len(chunks) * (400 / 1000) * 0.02)
    # print(chunks[20])
    # print('first chunk:', chunks[0])
    # return  # UNCOMMENT to check token count and price estimate before sending

    ###### Using Parallel
    # from joblib import Parallel, delayed

    # def handle_summary(summary):
    #     """Helper function to handle different data types"""
    #     if isinstance(summary, (list, str)):
    #         if isinstance(summary, list):
    #             for s in summary:
    #                 summaries.append(s)
    #         else:
    #             summaries.append(summary)

    # # Set the number of parallel jobs
    # n_jobs = 4

    # # Initialize the summaries list
    # summaries = []

    # # Use the Parallel function to parallelize the loop
    # Parallel(n_jobs=-1, backend="threading")(
    #     map(delayed(handle_summary), generate_summary(chunk))
    #     for chunk in tqdm(chunks, total=len(chunks))
    # )

    import concurrent.futures
    import time

    ###### Using ThreadPoolExecutor
    # # Set the maximum number of threads
    # max_threads = 4
    # # Initialize the summaries list
    # summaries = []
    # # Initialize the ThreadPoolExecutor
    # with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
    #     # Create a list of tasks
    #     tasks = [executor.submit(generate_summary, chunk) for chunk in chunks]
    #     # Wait for the tasks to complete
    #     for task in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
    #         # Get the result of the task
    #         summary = task.result()
    #         # Handle the result
    #         if isinstance(summary, (list, str)):
    #             if isinstance(summary, list):
    #                 for s in summary:
    #                     summaries.append(s)
    #             else:
    #                 summaries.append(summary)
    # # Merge the summaries into one
    # full_summary = "\n\n".join(summaries)
    ###### Using ThreadPoolExecutor but rate limited =)
    # Set the rate limit in seconds
    rate_limit = 1 / 25
    # Set the maximum number of threads
    max_threads = 2
    # Initialize the summaries list
    summaries = []
    # Initialize the ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        # Create a list of tasks
        tasks = []
        task_count = 0
        start_time = time.perf_counter()
        for chunk in chunks:
            # Submit a task to the executor
            task = executor.submit(generate_summary, chunk)
            # Append the task to the tasks list
            tasks.append(task)
            # Increment the task count
            task_count += 1
            # Calculate the elapsed time
            elapsed_time = time.perf_counter() - start_time
            # Print the task count and elapsed time
            print(f"Task count: {task_count}, Elapsed time: {elapsed_time:.2f} seconds")
            # Sleep for the rate limit
            time.sleep(rate_limit)
        end_time = time.perf_counter()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        # Print the elapsed time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        # Wait for the tasks to complete
        for task in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks)):
            # Get the result of the task
            summary = task.result()
            # Handle the result
            if isinstance(summary, (list, str)):
                if isinstance(summary, list):
                    for s in summary:
                        summaries.append(s)
                else:
                    summaries.append(summary)
    # Merge the summaries into one
    full_summary = "\n\n".join(summaries)
    ###### Using a for loop

    # summaries = []
    # for idx, chunk in tqdm(enumerate(chunks[:2]), total=len(chunks[:2])):
    #     # print(idx)
    #     # print(len(chunk))

    #     summary = generate_summary(chunk)
    #     if type(summary) == list:
    #         for s in summary:
    #             summaries.append(s)
    #     else:
    #         summaries.append(summary)

    # # Merge the summaries into one
    # full_summary = "\n\n".join(summaries)

    # Open the output file in write mode
    with open("output.txt", "w") as out_file:
        # Write the summary to the output file
        out_file.write(full_summary)


import textwrap

from transformers import GPT2TokenizerFast

# def split_text_to_chunks():
#     # Initialize the chunks list
#     chunks = []

#     # Load the tokenizer
#     tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

#     # Split the text into tokens using the tokenizer
#     tokens = tokenizer.encode(text, return_tensors=None)
#     print("len:", len(tokens))
#     # Split the tokens into chunks of 1340 tokens
#     for i in range(0, len(tokens), 1000):
#         chunk = tokens[i : i + 1000]
#         chunks.append(chunk)

#     return chunks


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

from typing import List


def split_text_to_chunks(text: str, max_tokens: int) -> List[str]:
    # Initialize the chunks list
    chunks = []

    # Split the text into tokens
    tokens = tokenizer.encode(text, return_tensors=None)

    # Initialize the start index
    start_index = 0

    # Iterate over the tokens
    while start_index < len(tokens):
        # Get the end index
        end_index = min(start_index + max_tokens, len(tokens))

        # Get the chunk of tokens
        chunk_tokens = tokens[start_index:end_index]

        # Convert the tokens to a string
        chunk_text = tokenizer.decode(
            chunk_tokens, truncation=True, skip_special_tokens=True
        )

        # Append the chunk to the chunks list
        chunks.append(chunk_text)

        # Set the start index to the end index
        start_index = end_index

    return chunks


import openai

from api_key import key

# Set the API key
openai.api_key = key


def generate_summary(text):
    max_tokens = 1500
    # Set the API endpoint
    model_engine = "text-davinci-003"

    # Set the prompt for the GPT-3 model
    prompt = f"Sumariza o texto em pt-pt:\n{text}\n\n"
    # print(prompt)

    # Make the API call
    try:
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            n=1,
            stop=None,
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1,
        )
        # Get the summary from the response
        summary = response["choices"][0]["text"].strip()
        print(response["choices"][0]["text"].strip(), "---------------")
        # Remove the prompt from the summary
        summary = summary.replace(prompt, "")

        print("ok")
        return summary
    except Exception as e:
        print("error")
        print(e)
        # Get the length of the string
        length = len(text)

        # Split the string in half
        half = length // 2

        # Get the first half of the string
        first_half = text[:half]

        # Get the second half of the string
        second_half = text[half:]
        summaries = []
        for i in range(2):
            if i == 0:
                prompt = f"Sumariza o texto em pt-pt:\n{first_half}\n\n"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    n=1,
                    stop=None,
                    temperature=0.5,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=1,
                    presence_penalty=1,
                )
                # Get the summary from the response
                summary = response["choices"][0]["text"].strip()

                # Remove the prompt from the summary
                summary = summary.replace(prompt, "")
                summaries.append(summary)
            else:
                prompt = f"Sumariza o texto em pt-pt:\n{second_half}\n\n"
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt,
                    n=1,
                    stop=None,
                    temperature=0.5,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=1,
                    presence_penalty=1,
                )
                # Get the summary from the response
                summary = response["choices"][0]["text"].strip()

                # Remove the prompt from the summary
                summary = summary.replace(prompt, "")
                summaries.append(summary)

            return summaries


# make a while(chunks > 4)
convert_pdf_to_txt(1)
# convert_pdf_to_txt(2) #This will summarize further, but needs to be in automated..
