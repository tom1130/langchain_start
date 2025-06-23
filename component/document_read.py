'''
document loading and summarization component
'''
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import HNLoader
from langchain.document_loaders import WikipediaLoader


def summary_hacker_news(post_id: str) -> str:
    
    google_api_key = os.getenv("gemini_api_key")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    
    client = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,  # Set temperature to 0 for deterministic output
        max_output_tokens=1024,  # Set maximum output tokens
        api_key=google_api_key
    )
    
    # Load Hacker News data
    post_url = f"https://news.ycombinator.com/item?id={post_id}"
    loader = HNLoader(post_url)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No documents found for post ID: {post_id}")
    
    texts = documents[0].page_content
    
    human_template = (
        '다음 Hacker News 게시물의 내용을 요약해 주세요:\n\n{text}'
    )
    
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [human_prompt]
    )
    print(f'texts: {texts}')
    print()
    print()
    messages = chat_prompt.format_messages(text=texts)
    response = client.invoke(messages)
    
    return response
    
def answer_wikipedia(person_name: str, question: str) -> str:
    """
    Answer a question about a person using Wikipedia.
    """
    google_api_key = os.getenv("gemini_api_key")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    
    client = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,  # Set temperature to 0 for deterministic output
        max_output_tokens=1024,  # Set maximum output tokens
        api_key=google_api_key
    )
    
    loader = WikipediaLoader(
        person_name,
        load_max_docs=1
    )
    documents = loader.load()
    context_text = documents[0].page_content if documents else ""
    print(context_text)
    human_template = (
        "질문에 답변해주세요:\n\n{question}\n\n{context}"
    )
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [human_prompt] 
    )
    chat = chat_prompt.format_prompt(question=question, context=context_text).to_messages()
    print(f'chat: {chat}')
    print()
    answer = client(chat)
    return answer
    

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Example usage
    # post_id = "12345"  # Replace with a valid Hacker News post ID
    # summary = summary_hacker_news(post_id)
    # print(summary)
    
    person_name = "Albert Einstein"  # Replace with a valid person's name
    question = "What is his contribution to physics?"
    answer = answer_wikipedia(person_name, question)
    print(answer)