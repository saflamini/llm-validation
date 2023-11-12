from pydantic import BaseModel
import assemblyai as aai
from assemblyai import LemurQuestionAnswer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import requests
nltk.download('punkt')
import os

# Replace with your API token
aai.settings.api_key = os.environ.get("assemblyai_key")
lemur_endpoint = "https://api.assemblyai.com/lemur/v3/generate/task"
headers = {
    "Authorization": os.environ.get("assemblyai_key")
}


# should use this for action items
transcriber = aai.Transcriber()

#comment back in to re-generate the transcript and action items w a new context
# transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/meeting.mp4")

# if transcript.error:
#   print(transcript.error)

# result = transcript.lemur.action_items(
#     context="A GitLab meeting to discuss logistics",
#     answer_format="**<topic header>**\n<relevant action items>\n",
# )

# print(result.response)

action_items_response = """
Here are the action items I would suggest based on the transcript:

**Metrics and KPIs**
- Update R&D wider MR rate KPI to show percentage of total MRs from community over time instead of MRs per external contributor.
- Add new KPI for average age of open bugs to supplement bug SLO metrics. 
- Update bug SLO metrics to measure percentage of open bugs within SLO timeframes rather than just closed bugs.
- Investigate adding security metrics to reflect amount of security work being prioritized.

**Database Performance**
- Create dedicated database server for data team to improve replication lag.
- Work with infrastructure team on database tuning improvements to reduce load.
- Partner with infrastructure team to optimize database performance as demand increases.

**Bug Backlog**
- Analyze data to understand spike in S2 bug close times.
- Prioritize clearing S3 and S4 bug backlog to improve SLOs.

**Productivity**
- Monitor developer productivity in March to see if MR rate rebounds as expected after seasonal dip.
"""


# Splitting the text into sections based on headers
sections = {}
current_section = None
for line in action_items_response.strip().split("\n"):
    if line.startswith("**") and line.endswith("**"):
        current_section = line[2:-2].strip()  # Remove the ** and strip any spaces
        sections[current_section] = []
    elif current_section and line.strip() != "":
        sections[current_section].append(line.strip())

# Convert the dictionary format into a standard list of strings

formatted_action_items_sections = []

for header, tasks in sections.items():
    section_text = header + "\n" + "\n".join(tasks)
    formatted_action_items_sections.append(section_text)

#this is the gitlab meeting transcript
transcript = aai.Transcript.get_by_id("6v4muko96g-2d7a-4bc9-883f-fb33b4691a8e")

paragraphs = []
for paragraph in transcript.get_paragraphs():
    paragraphs.append(paragraph.text)

sentences = []
for sentence in transcript.get_sentences():
    sentences.append(sentence.text)

def sliding_window(sentences, window_size=3):
    """
    Splits sentences into chunks using a sliding window approach.
    
    Args:
    - sentences (list of str): List of sentences from the transcript.
    - window_size (int): Number of sentences per chunk.
    
    Returns:
    - list of str: Chunks of sentences.
    """
    chunks = []
    for i in range(0, len(sentences) - window_size + 1):
        chunk = " ".join(sentences[i:i+window_size])
        chunks.append(chunk)
    return chunks


# Load the pre-trained sentence transformer model
# Note - the first time you do this, it will take some time to download the model
# We are downloading the model locally
# Why all-MiniLM-L6-v2? it's near the top of the leaderboard for semantic similarity search and is 5x faster than the largest model
# It's also a good model for semantic similarity because it's been trained on the STSB benchmark dataset
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer("infgrad/stella-base-en-v2")

# Embed each sentence in sentences
sentence_embeddings = model.encode(sentences)
paragraph_embeddings = model.encode(paragraphs)
# Create chunks of 3 sentences using the sliding window approach
sentence_chunks = sliding_window(sentences)
# Embed each chunk
chunk_embeddings = model.encode(sentence_chunks)

def compare_against_all_granularities(qa_embedding, embeddings_list, k):
    """
    Compare the action items embedding against a list of embeddings and return the top_k similar items' indices and scores.
    """
    similarities = cosine_similarity([qa_embedding], embeddings_list)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return top_k_indices, similarities[top_k_indices]


# Embed each action items and store a mapping between the action items text and the embedding
action_items_embeddings = {}
for action_item in formatted_action_items_sections:
    embedding = model.encode([action_item])[0]
    action_items_embeddings[action_item] = embedding


k = 3  # top k similar items to retrieve

new_action_items = []
filtered_action_items = []

sentence_threshold = 0.80 # Typically set higher than paragraph_threshold
paragraph_threshold = 0.73 # Typically set lower than sentence_threshold
chunk_threshold = 0.78  # This could be set between the sentence and paragraph thresholds

for action_item, action_item_embedding in action_items_embeddings.items():
    # Compare the summary sentence against all three granularities
    top_k_sentence_indices, top_k_sentence_similarities = compare_against_all_granularities(action_item_embedding, sentence_embeddings, k)
    top_k_paragraph_indices, top_k_paragraph_similarities = compare_against_all_granularities(action_item_embedding, paragraph_embeddings, k)
    top_k_chunk_indices, top_k_chunk_similarities = compare_against_all_granularities(action_item_embedding, chunk_embeddings, k)
    
    # Extract the most similar item from each granularity
    most_similar_sentence = sentences[top_k_sentence_indices[0]]
    most_similar_paragraph = paragraphs[top_k_paragraph_indices[0]]
    most_similar_chunk = sentence_chunks[top_k_chunk_indices[0]]
    
    # Check if the similarity scores cross their respective thresholds
    sentence_pass = any(sim >= sentence_threshold for sim in top_k_sentence_similarities)
    paragraph_pass = any(sim >= paragraph_threshold for sim in top_k_paragraph_similarities)
    chunk_pass = any(sim >= chunk_threshold for sim in top_k_chunk_similarities)
    
    # If any of the granularities pass their threshold, include the summary sentence
    if sentence_pass or paragraph_pass or chunk_pass:
        new_action_items.append(action_item)
    else:
        filtered_action_items.append(action_item)
    
    # Log the most similar items with their scores
    print("********************************")
    print(f"ACTION ITEM: {action_item}")
    print(f"Most similar sentence: {most_similar_sentence} (score: {top_k_sentence_similarities[0]})")
    print(f"Most similar paragraph: {most_similar_paragraph} (score: {top_k_paragraph_similarities[0]})")
    print(f"Most similar chunk: {most_similar_chunk} (score: {top_k_chunk_similarities[0]})")

print("***************************")
print("NEW ACTION ITEMS OUTPUT")
print(new_action_items)
print("***************************")
print("FILTERED ACTION ITEMS")
print(filtered_action_items)