import os 
from dotenv import load_dotenv
load_dotenv()

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
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

summary_chain = PROMPT | llm | StrOutputParser() | RunnableLambda(_strip_thinking)

# public function
def summarize(raw_text):
    return summary_chain.invoke({"text": raw_text[:6000]})