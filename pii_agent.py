from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from custom_wrapper import OpenRouterChat
from pydantic import BaseModel, Field
from typing import List
import os
import json

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class PIIOutput(BaseModel):
    pii_entities: List[str] = Field(default_factory=list)
    pii_types: List[str] = Field(default_factory=list)

llm = OpenRouterChat(
    api_key=OPENROUTER_API_KEY,
    model="openai/gpt-3.5-turbo",
    temperature=0,
    max_tokens=1024
)

parser = PydanticOutputParser(pydantic_object=PIIOutput)

prompt = ChatPromptTemplate.from_template("""
You are an expert PII extraction system.
Identify and extract all PII (Personally Identifiable Information) and PHIfrom the following input.

Return a JSON with two fields:
- pii_entities: list of extracted strings
- pii_types: matching type for each entity

Types to detect include:
Name, Email, Phone number, Address, SSN, Passport number, Credit card number,
Biometrics (face, fingerprint), Birthday, Age, Gender, Race, Location

Text:
{text}

Respond only with valid JSON as per the specified format.
""")

chain = (
    {"text": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

def extract_pii(text):
    try:
        result = chain.invoke(text)
        return result.dict()
    except Exception as e:
        print("Error during PII extraction:", e)
        return None

def format_text(txt_blocks):
    return "\n".join([item['text'] for item in txt_blocks])

def execute_pii(txt):
    input_text = format_text(txt)
    result = extract_pii(input_text)
    f=open("found_img/pii_text.json","w")
    json.dump(result,f,indent=2)
    f.close()
    return result

'''
p=open("found_img//text_loc.json","r")
txt=json.load(p)
p.close()
print(execute_pii(txt))'''
