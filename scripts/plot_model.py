import torch
from torchview import draw_graph
import os
import sys

# Add the parent directory of 'scripts' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Import custom models from model_manager
from model_manager import (
    get_answergeneration_model,
    get_questiongeneration_model,
)

# Define the path directory for saving plots
path_dir = r"D:\03PersonalFiles\Thesis\I-Comprehend\outputs\plots"

# Create the directory if it does not exist
if not os.path.exists(path_dir):
    os.makedirs(path_dir)

# Load the models and tokenizers
t5ag_model, t5ag_tokenizer = get_answergeneration_model()
t5qg_model, t5qg_tokenizer = get_questiongeneration_model()

def plot_ansgen_model(question, context, model, tokenizer, path_dir, difficulty):
    """Generate an answer for a given question and context. Optionally visualize attention weights."""
    
    # Combine question and context into the input format required by T5
    input_text = f"question: {question} context: {context}"
    
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    
    # Prepare encoder inputs
    encoder_input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)
    
    # Prepare decoder inputs (use padding token ID or dummy input)
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])
    
    # Generate the computation graph
    model_graph = draw_graph(
        model, 
        input_data={
            'input_ids': encoder_input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids
        }
    ).visual_graph
    
    # Save the graph to the specified directory
    output_file = os.path.join(path_dir, f"ansgen_{difficulty}.png")
    print(f"Saving Answer Generation Graph to: {output_file}")

    model_graph.render(directory=output_file, format='png')

    return output_file

def plot_quesgen_model(context, answer, model, tokenizer, path_dir, difficulty):
    """Generate a question for the given answer and context. Optionally visualize attention weights."""
    
    # Combine context and answer into the input format required by T5
    answer_span = context.replace(answer, f"<hl>{answer}<hl>", 1) + "</s>"
    
    # Tokenize input text
    inputs = tokenizer(answer_span, return_tensors="pt", padding=True, truncation=True)
    
    # Prepare encoder inputs
    encoder_input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)
    
    # Prepare decoder inputs (use padding token ID or dummy input)
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])
    
    # Generate the computation graph
    model_graph = draw_graph(
        model, 
        input_data={
            'input_ids': encoder_input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids
        }
    ).visual_graph
    
    # Save the graph to the specified directory
    output_file = os.path.join(path_dir, f"quesgen_{difficulty}.png")
    print(f"Saving Question Generation Graph to: {output_file}")

    model_graph.render(directory=output_file, format='png')

    return output_file

def generate_graphs_for_difficulty(difficulty_data):
    """Generate both question and answer generation graphs for a given difficulty level."""
    # Unpack difficulty data
    context = difficulty_data['context']
    question = difficulty_data['question']
    answer = difficulty_data['answer']
    
    # Generate the answer generation plot
    ansgen_output = plot_ansgen_model(question, context, t5ag_model, t5ag_tokenizer, path_dir, difficulty_data['difficulty'])
    print(f"Answer Generation Graph saved at: {ansgen_output}")
    
    # Generate the question generation plot
    quesgen_output = plot_quesgen_model(context, answer, t5qg_model, t5qg_tokenizer, path_dir, difficulty_data['difficulty'])
    print(f"Question Generation Graph saved at: {quesgen_output}")

if __name__ == "__main__":
    # Define difficulty levels with corresponding data
    easy = {'context': "The Eiffel Tower is located in Paris, France.",
            'question': "Where is the Eiffel Tower located?",
            'answer': "Paris, France", 'difficulty': 'easy'}
    
    medium = {'context': "Albert Einstein developed the theory of relativity, which revolutionized our understanding of space, time, and energy. He was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
              'question': "What did Albert Einstein win the Nobel Prize for in 1921?",
              'answer': "Explanation of the photoelectric effect", 'difficulty': 'medium'}
    
    hard = {'context': "The Treaty of Versailles was signed on June 28, 1919, marking the end of World War I. The treaty imposed heavy reparations and territorial losses on Germany, which contributed to economic hardship and political instability in the country, eventually leading to the rise of Adolf Hitler and the Nazi Party.",
            'question': "How did the Treaty of Versailles contribute to the rise of Adolf Hitler?",
            'answer': "The treaty imposed heavy reparations and territorial losses on Germany, causing economic hardship and political instability, which contributed to the rise of Adolf Hitler and the Nazi Party.", 'difficulty': 'hard'}

    # List of difficulty levels
    difficulty_levels = [easy, medium, hard]

    # Generate graphs for all difficulty levels
    for difficulty_data in difficulty_levels:
        generate_graphs_for_difficulty(difficulty_data)
