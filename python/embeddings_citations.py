from pydantic import BaseModel
import assemblyai as aai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import requests
nltk.download('punkt')

# Replace with your API token
aai.settings.api_key = ""
lemur_endpoint = "https://api.assemblyai.com/lemur/v3/generate/task"
headers = {
    "Authorization": ""
}

# URL of the file to transcribe
FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

# comment these two lines back in if you want to transcribe a new file
# transcriber = aai.Transcriber()
# transcript = transcriber.transcribe(FILE_URL)

#we have already transcribed this one - comment out this line if you're transcribing a new file
transcript = aai.Transcript.get_by_id("6mvfr2epvp-65bc-46dc-8b4e-b87487b5da4b")

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

def compare_against_all_granularities(summary_embedding, embeddings_list, k):
    """
    Compare the summary embedding against a list of embeddings and return the top_k similar items' indices and scores.
    """
    similarities = cosine_similarity([summary_embedding], embeddings_list)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return top_k_indices, similarities[top_k_indices]


lemur_summary = "Wildfires in Canada are making the air dirty in many places in the US. Smoke from the fires is traveling through the sky and making it hard to breathe in places like New York and Baltimore. The smoke has tiny pieces in it that can get inside your lungs if you breathe them. This can make you sick, especially kids and older adults. The pieces in the smoke are much more than normal and that's why the air is unhealthy. More people could get sick until the weather changes and moves the smoke away. Fires might happen more often in the future because of climate change, so dirty air could affect more places."

# Split the summary into sentences
# summary_sentences = lemur_summary.response.split(". ")
summary_sentences = nltk.sent_tokenize(lemur_summary)

# Embed each sentence in the summary and store a mapping between the sentence text and the embedding
summary_embeddings = {}
for sentence in summary_sentences:
    embedding = model.encode([sentence])[0]
    summary_embeddings[sentence] = embedding

k = 3  # top k similar items to retrieve

new_summary = ""
filtered_sentences = []

sentence_threshold = 0.80 # Typically set higher than paragraph_threshold
paragraph_threshold = 0.73 # Typically set lower than sentence_threshold
chunk_threshold = 0.78  # This could be set between the sentence and paragraph thresholds

for summary_sentence, summary_embedding in summary_embeddings.items():
    # Compare the summary sentence against all three granularities
    top_k_sentence_indices, top_k_sentence_similarities = compare_against_all_granularities(summary_embedding, sentence_embeddings, k)
    top_k_paragraph_indices, top_k_paragraph_similarities = compare_against_all_granularities(summary_embedding, paragraph_embeddings, k)
    top_k_chunk_indices, top_k_chunk_similarities = compare_against_all_granularities(summary_embedding, chunk_embeddings, k)
    
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
        new_summary += summary_sentence + " "
    else:
        filtered_sentences.append(summary_sentence)
    
    # Log the most similar items with their scores
    print("********************************")
    print(f"SUMMARY SENTENCE: {summary_sentence}")
    print(f"Most similar sentence: {most_similar_sentence} (score: {top_k_sentence_similarities[0]})")
    print(f"Most similar paragraph: {most_similar_paragraph} (score: {top_k_paragraph_similarities[0]})")
    print(f"Most similar chunk: {most_similar_chunk} (score: {top_k_chunk_similarities[0]})")

print("***************************")
print("NEW SUMMARY OUTPUT")
print(new_summary)
print("***************************")
print("FILTERED SENTENCES")
print(filtered_sentences)