from pydantic import BaseModel
import assemblyai as aai
from assemblyai import LemurQuestionAnswer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
import requests
nltk.download('punkt')

# Replace with your API token
assembly_key = os.environ.get("assemblyai_key")
aai.settings.api_key = assembly_key

lemur_endpoint = "https://api.assemblyai.com/lemur/v3/generate/task"
headers = {
    "Authorization": assembly_key
}

model = SentenceTransformer("infgrad/stella-base-en-v2")

def get_transcript(transcript_id, file_url):
    if transcript_id:
        return aai.Transcript.get_by_id(transcript_id)
    else:
        transcriber = aai.Transcriber()
        return transcriber.transcribe(file_url)

def extract_paragraphs_and_sentences(transcript):
    paragraphs = [p.text for p in transcript.get_paragraphs()]
    sentences = [s.text for s in transcript.get_sentences()]
    return paragraphs, sentences

def sliding_window(sentences, window_size=3):
    return [" ".join(sentences[i:i+window_size]) for i in range(0, len(sentences) - window_size + 1)]

def compare_against_all_granularities(embedding, embeddings_list, k):
    similarities = cosine_similarity([embedding], embeddings_list)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return top_k_indices, similarities[top_k_indices]

GRANULARITY_THRESHOLDS = {
    "sentence": 0.88,
    "paragraph": 0.80,
    "chunk": 0.84
}

def process_lemur_qa(lemur_qa, sentence_embeddings, paragraph_embeddings, chunk_embeddings, k=3):
    qa_embeddings = {}
    for item in lemur_qa:
        embedding = model.encode([item.answer])[0]
        qa_embeddings[item.answer] = embedding

    results = []
    for qa_item in lemur_qa:
        qa_answer = qa_item.answer
        qa_answer_embedding = qa_embeddings[qa_answer]

        top_k_sentence_indices, top_k_sentence_similarities = compare_against_all_granularities(qa_answer_embedding, sentence_embeddings, k)
        top_k_paragraph_indices, top_k_paragraph_similarities = compare_against_all_granularities(qa_answer_embedding, paragraph_embeddings, k)
        top_k_chunk_indices, top_k_chunk_similarities = compare_against_all_granularities(qa_answer_embedding, chunk_embeddings, k)

        # Calculate the margin by which each top similarity score exceeds its respective threshold
        sentence_margin = top_k_sentence_similarities[0] - GRANULARITY_THRESHOLDS["sentence"]
        paragraph_margin = top_k_paragraph_similarities[0] - GRANULARITY_THRESHOLDS["paragraph"]
        chunk_margin = top_k_chunk_similarities[0] - GRANULARITY_THRESHOLDS["chunk"]
        
        # Determine the granularity with the highest margin
        max_margin = max(sentence_margin, paragraph_margin, chunk_margin)
        
        if max_margin == sentence_margin and sentence_margin > 0:
            transcript_ref = sentences[top_k_sentence_indices[0]]
            most_similar_granularity = top_k_sentence_similarities[0]
        elif max_margin == paragraph_margin and paragraph_margin > 0:
            transcript_ref = paragraphs[top_k_paragraph_indices[0]]
            most_similar_granularity = top_k_paragraph_similarities[0]
        elif max_margin == chunk_margin and chunk_margin > 0:
            transcript_ref = sentence_chunks[top_k_chunk_indices[0]]
            most_similar_granularity = top_k_chunk_similarities[0]
        else:
            transcript_ref = "No reference met the threshold."
            most_similar_granularity = max_margin
        
        grounding_threshold_passed = max_margin > 0
        
        results.append({
            "question": qa_item.question,
            "answer": qa_answer,
            "citation": {
                "reference": transcript_ref,
                "similarity_score": most_similar_granularity
            },
            "grounding_threshold_passed": grounding_threshold_passed
        })
    return results


if __name__ == "__main__":
    API_KEY = ""
    FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"
    TRANSCRIPT_ID = "6vkxdgap5h-8d7b-4559-a702-16bf4c7c3b44"
    MODEL_NAME = "infgrad/stella-base-en-v2"

    # headers = initialize_assemblyai(API_KEY, "https://api.assemblyai.com/lemur/v3/generate/task")
    transcript = get_transcript(TRANSCRIPT_ID, FILE_URL)
    paragraphs, sentences = extract_paragraphs_and_sentences(transcript)
    sentence_chunks = sliding_window(sentences)
    sentence_embeddings = model.encode(sentences)
    paragraph_embeddings = model.encode(paragraphs)
    chunk_embeddings = model.encode(sentence_chunks)

    # Lemur QA processing
    lemur_qa = [LemurQuestionAnswer(question='What locations are affected by the wildfires?', answer='Maine, Maryland, Minnesota, New York City, the Mid Atlantic, and the Northeast.'), LemurQuestionAnswer(question='What is the cause of the wildfires?', answer='The wildfires are caused by dry conditions this season combined with weather systems channeling the smoke from the Canadian wildfires into parts of the US.'), LemurQuestionAnswer(question='Will the wildfires continue to proliferate?', answer='YES. The fires are expected to continue burning for a bit longer according to the expert.'), LemurQuestionAnswer(question='What is the relationship between climate change and wildfires?', answer='Climate change leads to an earlier start to fire season, fires lasting longer, and more frequent fires overall.'), LemurQuestionAnswer(question='Who is interviewed in the audio?', answer='Peter DiCarlo, an associate professor at Johns Hopkins University.')]

    results = process_lemur_qa(lemur_qa, sentence_embeddings, paragraph_embeddings, chunk_embeddings)
    for result in results:
        print(result)
        # question = result["question"]
        # answer = result["answer"]
        # reference = result["citation"]["reference"]
        # similarity = result["citation"]["similarity_score"]
    
        # print(f"QUESTION: {question}")
        # print(f"ANSWER: {answer}")
        # print(f"CITATION: {reference}")
        # print(f"SIMILARITY SCORE: {similarity}")
        print("*********************************")