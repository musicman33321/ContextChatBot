# cli.py

import os
import argparse
from dotenv import load_dotenv

# Load the core logic from your package!
from contextchatbot.chatbot import ContextualChatbot 

# Load environment variables at the start of the script
load_dotenv() 

def run_cli():
    """Defines the command-line interface for the ContextChatBot."""
    
    # 1. API Key Check
    if not os.getenv("OPENAI_API_KEY"):
         print("\nERROR: OPENAI_API_KEY is not set. Please create a .env file.")
         return
    
    # 2. Argument Parsing
    parser = argparse.ArgumentParser(
        description="Run a ContextChatBot query using fashion trend data.",
    )
    parser.add_argument(
        "question", 
        type=str, 
        help="The question to ask the chatbot (e.g., 'What size bags are fashionable in 2023?')."
    )
    
    args = parser.parse_args()
    
    # 3. Execution
    try:
        # data_path is relative to where the script is run (the root of your project)
        chatbot = ContextualChatbot(data_path='data/2023_fashion_trends.csv') 
        
        # Run query, obtaining original and context-augmented answerss
        num_tokens, user_query, answer_without_context, answer_with_context  = chatbot.answer_question(args.question)
        
        print(f"\nUser Query: {user_query}")
        print(f"\nAnswer Without Context:\n {answer_without_context}")
        print(f"\nAnswer With Context:\n {answer_with_context}")
        print(f"\n[Tokens Used: {num_tokens}]")
        
    except (ValueError, FileNotFoundError) as e:
        print(f"\nInitialization Error: {e}")
        print("Please ensure your data file and API key are correctly configured.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    run_cli()