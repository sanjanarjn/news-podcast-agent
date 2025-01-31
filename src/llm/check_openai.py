import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
# Initialize the OpenAI model using LangChain
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# Sample prompt
prompt = "What is the capital of France?"

# Use LangChain to invoke the model with the prompt
response = llm.invoke(prompt)

# Print the response
print("Response from OpenAI LLM:", response.content)