from pydantic import BaseModel
import assemblyai as aai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import requests
nltk.download('punkt')

# Replace with your API token
aai.settings.api_key = f"5c2f0f64b59d4e5fbdd677c9938a8e98"
lemur_endpoint = "https://api.assemblyai.com/lemur/v3/generate/task"
headers = {
    "Authorization": "5c2f0f64b59d4e5fbdd677c9938a8e98"
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

# Load the pre-trained sentence transformer model
# Note - the first time you do this, it will take some time to download the model
# We are downloading the model locally
# Why all-MiniLM-L6-v2? it's near the top of the leaderboard for semantic similarity search and is 5x faster than the largest model
# It's also a good model for semantic similarity because it's been trained on the STSB benchmark dataset
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Embed each sentence in sentences
sentence_embeddings = model.encode(sentences)

lemur_summary = transcript.lemur.summarize(
    context="Please summarize this transcript as though I were a 10 year old",
    answer_format="paragaraphs"
)

# Split the summary into sentences
# summary_sentences = lemur_summary.response.split(". ")
summary_sentences = nltk.sent_tokenize(lemur_summary.response)

# Embed each sentence in the summary and store a mapping between the sentence text and the embedding
summary_embeddings = {}
for sentence in summary_sentences:
    embedding = model.encode([sentence])[0]
    summary_embeddings[sentence] = embedding

k = 3 # top k similar sentences to retrieve

new_summary = ""
for summary_sentence, summary_embedding in summary_embeddings.items():
    similarities = cosine_similarity([summary_embedding], sentence_embeddings)[0]
    top_k_index = similarities.argsort()[-1]
    top_k_sentence = sentences[top_k_index]
    top_k_similarity = similarities[top_k_index]
    print(f"SUMMARY SENTENCE: {summary_sentence} \nMOST SIMILAR SENTENCE: {top_k_sentence} (similarity score: {top_k_similarity})\n")

def rebuild_summary_without_duplicates(summary_embeddings, sentence_embeddings, threshold=0.6, k=3):
    """
    Rebuilds the summary by filtering out potentially hallucinated sentences.
    """
    new_summary = ""
    added_sentences = set()
    filtered_sentences = []

    for summary_sentence, summary_embedding in summary_embeddings.items():
        similarities = cosine_similarity([summary_embedding], sentence_embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        top_k_similarities = [similarities[i] for i in top_k_indices]
        if any(sim >= threshold for sim in top_k_similarities) and summary_sentence not in added_sentences:
            new_summary += summary_sentence + " "
            added_sentences.add(summary_sentence)
        else:
            filtered_sentences.append(summary_sentence)

    return new_summary.strip(), filtered_sentences

# Using the provided embeddings and logic to rebuild the summary without duplicates
new_summary, filtered_sentences = rebuild_summary_without_duplicates(summary_embeddings, sentence_embeddings)
print("***************************")
print("NEW SUMMARY OUTPUT")
print(new_summary)
print("***************************")
print("FILTERED SENTENCES")
print(filtered_sentences)

