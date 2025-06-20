'''
format instructions : 모델에게 출력 형식을 설명하는 지침(string 값)
    - 예시 : 'Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`'(CommaSeparatedListOutputParser의 경우)
parser : 원하는 데이터 타입으로 변환하는 메소드 제공
    - output fixing parser : 형식 지침과 일치하지 않을 때 수정하기 위한 도구
pydantic : 파싱된 출력이 올바른지 확인하는 도구
'''
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import DatetimeOutputParser, OutputFixingParser, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.schema import HumanMessage, AIMessage
# parser 예제
def generate_datetime_output(prompt: str) -> str:
    """
    Generate a datetime output based on the provided prompt.
    """
    system_template = """
    당신은 항상 질문에 날짜와 시간 패턴에 맞게만 답변해야 합니다.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    human_prompt = HumanMessagePromptTemplate.from_template(
        "{user_input}\n\n{format_instructions}"    
    )
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt,
         human_prompt]
    )
    date_output_parser = DatetimeOutputParser()
    date_format_instructions = date_output_parser.get_format_instructions()
    chat = chat_prompt.format_messages(
        user_input=prompt,
        format_instructions=date_format_instructions
        )
    
    # Get the Google API key from environment variables
    google_api_key = os.getenv("gemini_api_key")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    
    client = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0, # Set temperature to 0 for deterministic output
        max_output_tokens=1024, # Set maximum output tokens
        api_key=google_api_key
    )
    
    response = client.invoke(chat)
    try:
        # Parse the response using the datetime output parser
        parsed_response = date_output_parser.parse(response.content)
        return parsed_response
    except Exception as e:
        print("Error parsing response:")
        print(e)
    
    fixing_parser = OutputFixingParser.from_llm(parser=date_output_parser, llm=client)
    try:
        # Attempt to fix the output using the fixing parser
        fixed_response = fixing_parser.parse(response.content)
        return fixed_response
    except Exception as e:
        print("Error fixing response:")
        print(e)
    
    
# pydantic 예제
class Scientist(BaseModel):
    name: str = Field(description="The name of the scientist")
    field: str = Field(description="The field of expertise")
    discovery: str = Field(description="The discovery made by the scientist")

def generate_scientist_info(prompt: str) -> Scientist:
    
    output_parser = PydanticOutputParser(pydantic_object=Scientist)
    format_instructions = output_parser.get_format_instructions()
    
    human_prompt = HumanMessagePromptTemplate.from_template(
        "{user_input}\n\n{format_instructions}"
    )
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [human_prompt]
    )
    chat = chat_prompt.format_messages(
        user_input=prompt,
        format_instructions=format_instructions
    )
    
    # Get the Google API key from environment variables
    google_api_key = os.getenv("gemini_api_key")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    
    client = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0, # Set temperature to 0 for deterministic output
        max_output_tokens=1024, # Set maximum output tokens
        api_key=google_api_key
    )
    
    response = client.invoke(chat)
    print(format_instructions)
    parsed_response = output_parser.parse(response.content)
    return parsed_response
    

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Example usage
    prompt = "13차 수정헌법이 언제 비준되었나요?"
    response = generate_datetime_output(prompt)
    print(response)
    
    prompt = "안토니오 가우디에 대해 알려주세요"
    response = generate_scientist_info(prompt)
    print(response.name)
    print(response.field)
    print(response.discovery)
    print(response)