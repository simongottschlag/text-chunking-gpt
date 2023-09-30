# Converting raw text chunks to Markdown using GPT
This is an example project to levrage GPT to convert raw text chunks to markdown using GPT.

We are using GPT function call to drive a loop and inject raw text chunks based on what GPT is requesting.

# Running it

Create the file `.tmp/env` and add the following:

    AZURE_OPENAI_KEY=[key]
    AZURE_OPENAI_ENDPOINT=[endpoint]

Make sure you have a deployment named `gpt-4` in the environment.

Run it using: `make run`