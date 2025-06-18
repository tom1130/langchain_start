import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

def generate_trip_plan(destination: str, budget: int, duration: int) -> str:
    """
    Generate a trip plan based on the provided destination, budget, and duration.
    """
    # Define the system prompt
    system_prompt = (
        "You are a travel planner. Your task is to create a detailed trip plan "
        "user's destination is {destination}, budget is {budget} USD, and duration is {duration} days."
    )
    system_template = SystemMessagePromptTemplate.from_template(system_prompt)
    
    # Define the human prompt
    human_prompt = (
        "Create a detailed trip plan for the user. "
        "Include recommendations for flights, accommodations, activities, and dining options. "
        "Make sure to stay within the budget and consider the duration of the trip."
    )
    human_template = HumanMessagePromptTemplate.from_template(human_prompt)
    
    # Create the chat prompt template
    chat_template = ChatPromptTemplate.from_messages(
        [system_template, human_template]
    )
    
    formatted_prompt = chat_template.format_messages(
        destination=destination,
        budget=budget,
        duration=duration
    )
    
    client = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0, # Set temperature to 0 for deterministic output
        max_output_tokens=1024, # Set maximum output tokens
        api_key=google_api_key
    )
    
    # Generate the trip plan using the Google Generative AI client
    response = client(formatted_prompt)
    return response.content

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the Google API key from environment variables
    google_api_key = os.getenv("gemini_api_key")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    # Example usage
    destination = "Paris"
    budget = 2000  # USD
    duration = 7  # days
    trip_plan = generate_trip_plan(destination, budget, duration)
    print(trip_plan)