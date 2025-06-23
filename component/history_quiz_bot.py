import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.output_parsers import DatetimeOutputParser

class HistoryQuizBot:
    
    def __init__(self):
        load_dotenv()
        self.google_api_key = os.getenv("gemini_api_key")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        self.client = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0,  # Set temperature to 0 for deterministic output
            max_output_tokens=1024,  # Set maximum output tokens
            api_key=self.google_api_key
        )
        
    def generate_question(self, topic: str) -> str:
        """
        Generate a history quiz question based on the provided topic.
        """
        system_template = (
            "You are a history quiz bot. "
            "Your task is to generate a quiz question based on the provided topic. "
            "the answer is correct date about {topic}"
            "do not suggest the answer, just generate the question."
        )
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        human_template = (
            "Generate a quiz question about {topic}.\n"
            "Make sure the question is clear and concise."
        )
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt]
        )
        formatted_prompt = chat_prompt.format_messages(topic=topic)
        
        response = self.client.invoke(formatted_prompt)
        question = response.content.strip()
        return question
    
    def get_answer(self, question: str) -> datetime:
        """
        Get the answer to the quiz question in datetime format.
        """
        system_template = (
            "You are a expert who announce date of the historic event. "
            "Your task is to provide the date in a specific format: YYYY-MM-DD. "
        )
        
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        human_template = (
            "Given the question: {question}, "
            "\n\n{format_instructions}"  # This will be replaced with format instructions
        )
        
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt]
        )
        datetime_output_parser = DatetimeOutputParser()
        format_instructions = datetime_output_parser.get_format_instructions()
        chat = chat_prompt.format_messages(question=question, format_instructions=format_instructions)
        
        response = self.client.invoke(chat)
        try:
            # Parse the response using the datetime output parser
            parsed_response = datetime_output_parser.parse(response.content)
            return parsed_response
        except Exception as e:
            print("Error parsing response:")
            print(e)
            return None
        
    def get_user_response(self, question: str) -> str:
        """
        Get the user's response to the quiz question.
        """
        print('Please type your answer in the format YYYY-MM-DD.')
        user_response = input(f"Your answer for the question '{question}': ")
        
        try:
            # Attempt to parse the user response as a datetime
            parsed_date = datetime.strptime(user_response, "%Y-%m-%d")
            return parsed_date
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            return self.get_user_response(question)
        
    def check_answer(self, user_response: datetime, correct_answer: datetime) -> bool:
        """
        Check if the user's response matches the correct answer.
        """
        if user_response == correct_answer:
            print("Correct! Well done.")
            return True
        else:
            print(f"Incorrect. The correct answer is {correct_answer.strftime('%Y-%m-%d')}.")
            return False
        
if __name__ == "__main__":
    bot = HistoryQuizBot()
    topic = "The French Revolution"
    
    question = bot.generate_question(topic)
    print(f"Generated Question: {question}")
    
    user_response = bot.get_user_response(question)
    print(f"Your Response: {user_response.strftime('%Y-%m-%d')}")
    
    answer = bot.get_answer(question)
    print(f"Answer: {answer.strftime('%Y-%m-%d')}")

    is_correct = bot.check_answer(user_response, answer)