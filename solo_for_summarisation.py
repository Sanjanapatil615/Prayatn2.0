import json
import solo
from torch.utils.data import DataLoader, Dataset

# Define a Dataset class
class ComplaintDataset(Dataset):
    def _init_(self, complaints, max_length=1024):
        self.complaints = complaints
        self.max_length = max_length

    def _len_(self):
        return len(self.complaints)

    def _getitem_(self, idx):
        complaint = self.complaints[idx]
        title = complaint.get('title', '')
        text = complaint.get('text', '')
        input_text = f"{title}. {text}"
        return {
            'text': input_text,
            'url': complaint.get('url', 'No URL'),
            'id': complaint.get('id', idx + 1)
        }

# Summarization function
def summarize_batch(batch, model):
    summaries = []
    for complaint in batch:
        prompt = f"Summarize the following complaint: {complaint['text']}"
        summary = model.generate(prompt)
        summaries.append({
            'id': complaint['id'],
            'summary': summary
        })
    return summaries

# Main function to process complaints

def process_complaints(input_json_path, output_json_path, model_path, batch_size=2):
    # Load Qwen Model
    model = solo.load_local(model_path)
    
    # Load complaints from JSON file
    with open(input_json_path, 'r', encoding='utf-8') as file:
        complaints = json.load(file)
    
    # Create Dataset and DataLoader
    dataset = ComplaintDataset(complaints, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    processed_complaints = []
    
    # Iterate over batches and process
    for batch in dataloader:
        summaries = summarize_batch(batch, model)
        processed_complaints.extend(summaries)
    
    # Save processed complaints to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(processed_complaints, output_file, ensure_ascii=False, indent=4)

# Example usage
if _name_ == "_main_":
    input_json_path = 'complaint_ps.json'  # Path to input JSON file
    output_json_path = 'complaint.json'  # Path to output JSON file
    model_path = "C:/users/pujwal/solo/qwen"  # Update this with the actual model path
    process_complaints(input_json_path, output_json_path, model_path)
