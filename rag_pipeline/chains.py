import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=3
)

# Prompt
PROMPT = ChatPromptTemplate.from_template("""
You are a helpful research assistant. Read the paper excerpt and return a structures summary.
                                          
Paper:
{text}
                                          
Respond in exactly this format:

## Main Contributions
- (3-5 bullet points on key findings and novelty)

## Architecture & Methods
- (model design, datasets used, techniques)
                                          
## Key Results
- (quantitative performance, comparisons to baselines)

## Limitations & Future Work
- (weaknesses or gaps mentioned by authors)                                          
""")

summary_chain = PROMPT | llm | StrOutputParser()

# public function
def summarize(raw_text):
    return summary_chain.invoke({"text": raw_text[:6000]})