import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from keybert import KeyBERT

# Ensure SentencePiece is installed
try:
    import sentencepiece
except ImportError:
    raise ImportError(
        "T5Tokenizer requires the SentencePiece library but it was not found in your environment. "
        "Install it using 'pip install sentencepiece'."
    )

# Load T5 Model & Tokenizer
model_name = "t5-small"  # Use "t5-base" for better accuracy
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Summarization Function
def summarize_text(text, max_length=150, min_length=50):
    # Prepend 'summarize: ' as T5 requires task prefix
    input_text = "summarize: " + text
    
    # Tokenize Input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate Summary
    summary_ids = model.generate(**inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4)
    
    # Decode & Return Summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Define a Dataset class
class ComplaintDataset(Dataset):
    def _init_(self, complaints, tokenizer, max_length=1024):
        self.complaints = complaints
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _len_(self):
        return len(self.complaints)

    def _getitem_(self, idx):
        complaint = self.complaints[idx]
        title = complaint.get('title', '')
        text = complaint.get('description', '')  # Use 'description' instead of 'text'
        input_text = f"{title}. {text}"
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'url': complaint.get('url', 'No URL'),
            'text': input_text
        }

# Summarization function with T5
def summarize_batch(batch, model, tokenizer, device, max_length=150, min_length=50):
    model.eval()
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=2,  # Reduced number of beams
            early_stopping=True
        )
        summaries = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in summary_ids
        ]
    return summaries

# Keyword extraction function using KeyBERT
def extract_keywords(text, kw_model, num_keywords=5):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    return [kw[0] for kw in keywords]

# Main function to process complaints
def process_complaints(input_json_path, output_json_path, batch_size=1):  # Further reduced batch size
    # Load KeyBERT model for keyword extraction
    kw_model = KeyBERT()

    # Check if GPU is available and move model to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load complaints from JSON file
    with open(input_json_path, 'r', encoding='utf-8') as file:
        complaints = json.load(file)

    # Create Dataset and DataLoader
    dataset = ComplaintDataset(complaints, tokenizer, max_length=512)  # Reduced max length
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # Use more workers

    processed_complaints = []

    # Iterate over batches and process
    for batch_idx, batch in enumerate(dataloader):
        summaries = summarize_batch(batch, model, tokenizer, device)
        for i, summary in enumerate(summaries):
            complaint_idx = batch_idx * batch_size + i + 1
            text = batch['text'][i]
            keywords = extract_keywords(text, kw_model)
            processed_complaints.append({
                'complaint_number': complaint_idx,
                'summary': summary,
                'keywords': keywords
            })

    # Save processed complaints to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as output_file:
        json.dump(processed_complaints, output_file, ensure_ascii=False, indent=4)

# Example usage
if __name__ == "__main__":
    input_json_path = 'india_complaints_reddit2.json'  # Path to input JSON file
    output_json_path = 'processed_complaints.json'  # Path to output JSON file
    process_complaints(input_json_path, output_json_path)

    # Test Summarization
    long_text = """Your long multilingual text here..."""
    print("Summary:", summarize_text(long_text))