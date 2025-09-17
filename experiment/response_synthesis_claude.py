import json
import time
import argparse
from anthropic import Anthropic
from typing import List, Dict
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# Initialize the Anthropic client with your API key
client = Anthropic(api_key='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')

def create_batch_requests(instruction_data: List[Dict]) -> List[Request]:
    """
    Create batch requests from instruction data.
    """
    batch_requests = []
    for i, item in enumerate(instruction_data):
        batch_request = Request(
            custom_id=str(i),
            params=MessageCreateParamsNonStreaming(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": item["instruction"]
                }],
                temperature=0.7
            )
        )
        batch_requests.append(batch_request)
    return batch_requests

def check_batch_status(batch_id: str):
    """
    Check the status of a batch job.
    """
    return client.messages.batches.retrieve(batch_id)

def process_batch_results(batch_id: str, instruction_data: List[Dict]) -> List[Dict]:
    """
    Process batch results and extract instruction-output pairs.
    
    Args:
        batch_id: ID of the batch job
        instruction_data: Original list of instruction dictionaries
        
    Returns:
        List of dictionaries containing instructions and outputs
    """
    synthetic_response = []
    
    # Create a dictionary to store original instructions
    instruction_dict = {}
    for i, item in enumerate(instruction_data):
        instruction_dict[str(i)] = item["instruction"]
    
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            # Get the instruction using custom_id
            instruction = instruction_dict[result.custom_id]
            output = result.result.message.content[0].text
            synthetic_response.append({
                "instruction": instruction,
                "output": output
            })
        else:
            print(f"Request {result.custom_id} failed with status: {result.result.type}")
    
    return synthetic_response

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='response')
    args = parser.parse_args()

    # Step 1: Load instruction data
    with open(f"../data/instruction/{args.mode}_instruction.json") as f:
        instruction_data = json.load(f)
    # Step 2: Create and submit batch job
    batch_requests = create_batch_requests(instruction_data)
    message_batch = client.messages.batches.create(requests=batch_requests)
    print(f"Batch job created. Batch ID: {message_batch.id}")

    # Step 3: Poll the status until completion
    while True:
        status = check_batch_status(message_batch.id)
        print(f"Current status: {status.processing_status}")
        print(f"Request counts: {status.request_counts}")
        
        if status.processing_status == "ended":
            print("Batch job completed.")
            break
        
        time.sleep(30)  # Wait for 30 seconds before checking again
    
    # Step 4: Process results and save
    synthetic_response = process_batch_results(message_batch.id, instruction_data)
    
    output_path = f"../data/synthetic_response/{args.mode}_synthetic_response_claude.json"
    with open(output_path, "w") as f:
        json.dump(synthetic_response, f, indent=2)
    print(f"All responses saved to {output_path}")

if __name__ == "__main__":
    main()