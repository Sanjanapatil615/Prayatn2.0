import os
import torch
import string
import shutil
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# -------------------------------
# Step 1: Define Complaint Categories & Labels
# -------------------------------

complaint_labels = [
    "Water & Sanitation Department",
    "Roads & Transport Department",
    "Electricity Department",
    "Public Safety & Police",
    "Healthcare Services",
    "Education Department",
    "Municipal Corporation",
    "Taxation & Finance Department"
]

# -------------------------------
# Step 2: Prepare Training Data
# -------------------------------

# 20 complaints for each department as a typical middle-class urban citizen
water_complaints = [
    "Intermittent water supply disrupting daily chores.",
    "Very low water pressure during morning and evening hours.",
    "Concerns over water contamination and quality issues.",
    "Repeated delays in fixing burst pipelines.",
    "Unhygienic conditions at public water taps in the neighborhood.",
    "Poor maintenance of water storage facilities in residential areas.",
    "Frequent waterlogging during the rainy season due to poor drainage.",
    "Lack of timely repair for leaking pipes in the locality.",
    "Absence of proper filtration systems in public distribution points.",
    "Erratic billing despite the irregularity of supply.",
    "Perceived corruption in the allocation of water resources.",
    "Faulty water meters leading to unexpectedly high bills.",
    "Inadequate communication about scheduled maintenance work.",
    "No clear alternative plan during extended outages.",
    "Mismanagement of sewage disposal causing foul odors.",
    "Slow response to registered consumer complaints.",
    "Water shortages intensifying during peak summer months.",
    "Unsafe storage practices leading to public health risks.",
    "Neglected repair work on visible leaks in communal areas.",
    "Insufficient investment in upgrading old water infrastructure."
]

road_complaints = [
    "Roads riddled with potholes causing vehicle damage.",
    "Delays in repairing road cracks and damages after monsoon.",
    "Inadequate street lighting on major and residential roads.",
    "Traffic congestion due to inefficient signal management.",
    "Frequent delays and cancellations of public buses.",
    "Absence of safe pedestrian crossings and sidewalks.",
    "Non-functional traffic signals at critical intersections.",
    "Overcrowded public transport during peak travel times.",
    "Poor connectivity to suburban areas affecting daily commute.",
    "Neglected maintenance of public transport vehicles.",
    "Unplanned road widening causing inconvenience to residents.",
    "Shortage of adequate parking spaces in busy areas.",
    "Lax enforcement of traffic rules leading to unsafe driving.",
    "Delayed execution of promised new road projects.",
    "Mismanagement of construction work causing extended disruptions.",
    "Inadequate road signage leading to frequent confusion.",
    "Frequent breakdowns of metro and local train services.",
    "Limited availability of eco-friendly transport options.",
    "Delays in implementing modern traffic management technologies.",
    "Suspected corruption in awarding road construction contracts."
]

electricity_complaints = [
    "Repeated power cuts that disrupt work and daily life.",
    "Unreliable electricity supply during peak hours.",
    "Slow response to restore power after outages.",
    "Inaccurate billing with inexplicable high charges.",
    "Unexplained voltage fluctuations damaging home appliances.",
    "Outdated electrical infrastructure causing regular issues.",
    "Lack of clear communication on scheduled load shedding.",
    "Inadequate emergency support during extended outages.",
    "Poor customer service and delayed complaint redressal.",
    "Sudden power surges causing appliance failures.",
    "Hidden charges and lack of transparency in billing.",
    "Difficulty in obtaining prompt meter replacement or repair.",
    "Frequent unannounced outages without any advisory.",
    "Delays in connection for new or relocated households.",
    "Inefficient integration of renewable energy sources.",
    "Overbilling issues without proper justification.",
    "Faulty meters that often result in disputed bills.",
    "Inconsistent application of regulations across regions.",
    "High costs associated with installation fees for new connections.",
    "Lack of timely updates during blackout events."
]

police_complaints = [
    "Slow police response during emergencies.",
    "Inadequate patrolling in residential areas at night.",
    "Scarcity of female officers compromising safety for women.",
    "Poor handling of petty crimes and local disturbances.",
    "Widespread corruption and instances of bribery.",
    "Insufficient enforcement of traffic regulations.",
    "Failure to address recurring community issues.",
    "Instances of unprofessional and rude behavior by officers.",
    "Lack of proactive community engagement initiatives.",
    "Insufficient training for handling protests and large gatherings.",
    "Delays in processing and following up on filed cases.",
    "Poor monitoring of known criminal hotspots.",
    "Inadequate coordination with other law enforcement agencies.",
    "Mismanagement during public events and demonstrations.",
    "Inconsistencies in maintaining law and order in different areas.",
    "Negligence in promptly addressing domestic violence cases.",
    "Lack of transparency in investigations and case handling.",
    "Overreliance on force rather than community dialogue.",
    "Inadequate measures to curb street-level crimes.",
    "Deficient surveillance in areas with higher crime rates."
]

health_complaints = [
    "Long waiting times at government hospitals and clinics.",
    "Shortage of essential medicines and medical supplies.",
    "Unavailability of specialized doctors in public facilities.",
    "Poor sanitation and hygiene in hospital wards.",
    "Delays in emergency services during peak times.",
    "Overcrowded outpatient departments leading to rushed care.",
    "Lack of proper patient care and attention from staff.",
    "Limited availability of advanced diagnostic equipment.",
    "Inadequate coordination between departments in hospitals.",
    "High risk of hospital-acquired infections.",
    "Poor management of patient records and histories.",
    "Absence of preventive healthcare and wellness programs.",
    "Insufficient mental health support and counseling services.",
    "Limited ambulance services leading to delayed response.",
    "Understaffed facilities affecting quality of care.",
    "Suspected corruption in billing and service charges.",
    "Unresponsive helpline services during crises.",
    "Delays in scheduling and performing critical surgeries.",
    "Inadequate communication regarding patient treatment plans.",
    "Insufficient investment in rural and underserved health centers."
]

education_complaints = [
    "Overcrowded classrooms reducing individual attention to students.",
    "Outdated teaching methods that do not engage modern learners.",
    "Poor infrastructure and dilapidated school buildings.",
    "Insufficient training programs for government school teachers.",
    "Lack of proper security measures on school premises.",
    "Low teacher salaries affecting morale and teaching quality.",
    "Limited integration of modern technology in classrooms.",
    "Uneven quality of education across different schools.",
    "Delays in implementing updated curricula and reforms.",
    "Inadequate funding for extracurricular and sports activities.",
    "Poor scheduling and management of examinations.",
    "Lack of remedial classes for students falling behind.",
    "Insufficient maintenance of school facilities like labs and libraries.",
    "Absence of career counseling and guidance services.",
    "Inability to address high rates of student absenteeism.",
    "Limited access to quality education in rural or peripheral areas.",
    "Suspicions of corruption in the admissions process.",
    "Poor sanitation and insufficient clean drinking water in schools.",
    "Lack of special education support for differently-abled students.",
    "Inconsistent evaluation and grading practices across schools."
]

municipal_complaints = [
    "Delays and irregularities in garbage collection schedules.",
    "Inadequate street cleaning leading to unhygienic public spaces.",
    "Poor upkeep of public parks, gardens, and recreational areas.",
    "Inconsistent inspection of food vendors and street markets.",
    "Slow response to resident complaints and public grievances.",
    "Inefficient urban planning leading to chaotic city layouts.",
    "Erratic water supply despite municipal promises.",
    "Neglect of maintenance work on local roads and footpaths.",
    "Protracted delays in launching promised infrastructure projects.",
    "Lack of regulation resulting in unauthorized encroachments.",
    "No clear policy on waste segregation and recycling initiatives.",
    "Insufficient budget allocation for essential civic amenities.",
    "Poor management of community facilities like sports grounds.",
    "Inadequate support for community policing and neighborhood watches.",
    "Delay in addressing illegal constructions in residential areas.",
    "Neglected maintenance of street lights affecting nighttime safety.",
    "Perceived corruption in awarding municipal service contracts.",
    "Slow enforcement of building safety and sanitation norms.",
    "Lack of community consultation for urban development projects.",
    "Overlooked issues in sewage management and drainage systems."
]

tax_complaints = [
    "Complex and confusing tax filing procedures for residents.",
    "Frequent errors in property tax assessments.",
    "Poor communication regarding sudden changes in tax rates.",
    "Lengthy processing times for tax refunds.",
    "Excessive paperwork required for routine tax submissions.",
    "Inadequate digital services making tax payments cumbersome.",
    "Lack of transparency in how tax revenues are utilized.",
    "Slow responses to inquiries related to tax matters.",
    "Perceived favoritism in granting tax concessions.",
    "High penalties imposed for minor filing errors.",
    "Mismanagement of local government funds impacting services.",
    "Delays in disbursing municipal finances to various departments.",
    "Inaccurate record-keeping of tax payments and receipts.",
    "Overly complex rules affecting small business owners.",
    "Unjustified tax hikes without clear public justification.",
    "Inadequate support and guidance for low-income taxpayers.",
    "Poor grievance redressal for disputed tax issues.",
    "Confusing online tax portals that are not user-friendly.",
    "Excessive focus on revenue collection over service delivery.",
    "Limited accountability and oversight in financial management."
]

# Combine all complaints and create labels accordingly.
all_complaints = (
    water_complaints +
    road_complaints +
    electricity_complaints +
    police_complaints +
    health_complaints +
    education_complaints +
    municipal_complaints +
    tax_complaints
)

# Generate labels: first 20 examples have label 0, next 20 label 1, and so on.
labels = []
for i in range(len(complaint_labels)):
    labels.extend([i] * 20)  # 20 complaints per department

train_data = {
    "text": all_complaints,
    "label": labels
}

dataset = Dataset.from_dict(train_data)

# -------------------------------
# Step 3: Preprocess & Tokenize Data
# -------------------------------

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def clean_text(text):
    """Lowercase text and remove punctuation."""
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def tokenize_function(examples):
    """Tokenize and clean text."""
    cleaned_texts = [clean_text(text) for text in examples["text"]]
    return tokenizer(cleaned_texts, truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# -------------------------------
# Step 4: Fine-Tune the Model
# -------------------------------

num_labels = len(complaint_labels)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    eval_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

try:
    trainer.train()
except Exception as e:
    print(f"Training error: {e}")

# -------------------------------
# Step 5: Save Trained Model
# -------------------------------

save_path = "./complaint_classification_model"

# Remove the folder if it exists to avoid conflicts
shutil.rmtree(save_path, ignore_errors=True)

try:
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")

# -------------------------------
# Step 6: Inference Function
# -------------------------------

def predict_complaint(text):
    """Predict the complaint category from text input."""
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return {
        "text": text,
        "predicted_category": complaint_labels[predicted_class]
    }

# -------------------------------
# Step 7: Process Complaints from JSON and Save Output
# -------------------------------

def process_complaints(input_json_path, output_json_path):
    """Process complaints from a JSON file and save categorized results."""
    # Load complaints from JSON file
    with open(input_json_path, 'r', encoding='utf-8') as file:
        complaints = json.load(file)

    categorized_complaints = []

    # Predict category for each complaint
    for complaint in complaints:
        description = complaint.get("description", "")
        predicted_result = predict_complaint(description)
        complaint["predicted_category"] = predicted_result["predicted_category"]
        categorized_complaints.append(complaint)

    # Save the categorized complaints to a new JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(categorized_complaints, f, ensure_ascii=False, indent=4)

    print(f"Categorized complaints saved to {output_json_path}")

# Example usage
if __name__ == "__main__":
    input_json_path = 'processed_complaints.json'  # Path to input JSON file
    output_json_path = 'categorized_complaints.json'  # Path to output JSON file
    process_complaints(input_json_path, output_json_path)