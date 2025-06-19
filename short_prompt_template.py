import os
import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.schema import HumanMessage, AIMessage

def generate_simple_text(prompt: str) -> str:
    """
    Generate a simple text response based on the provided law prompt
    """
    system_template = """
    당신은 유용한 법률 보조입니다. 복잡한 법률 용어를 쉽고 이해하기 쉽게 번역합니다.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    legal_text_example = """
    이 계약의 어떠한 조항도 무효, 불법 또는 집행 불가능한 것으로 판명되는 경우,
    해당 조항은 필요한 최소한의 범위 내에서 수정된 것으로 간주되며, 나머지 조항의 유효성은 영향을 받지 않습니다.
    """
    example_human_message = HumanMessage(content=legal_text_example)
    
    simplified_ai_example = """
    계약의 일부 조항이 무효가 되더라도, 나머지 조항은 여전히 유효합니다.
    """
    example_ai_message = AIMessage(content=simplified_ai_example)
    
    new_human_message = HumanMessagePromptTemplate.from_template("{legal_text}")
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt,
         example_human_message,
         example_ai_message,
         new_human_message
        ]
    )
    chat = chat_prompt.format_messages(legal_text=prompt)
    client = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0, # Set temperature to 0 for deterministic output
        max_output_tokens=1024, # Set maximum output tokens
        api_key=google_api_key
    )
    response = client.invoke(chat)
    return response.content

if __name__ == "__main__":
    dotenv.load_dotenv()
    
    # Get the Google API key from environment variables
    google_api_key = os.getenv("gemini_api_key")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    
    # Example usage
    prompt = """
    양도인은 본 계약에 따라 부동산의 소유권을 양수인에게 이전하며,
    이는 모든 권리와 의무를 포함합니다."""
    response = generate_simple_text(prompt)
    print(response)