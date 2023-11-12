import assemblyai as aai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

nltk.download('punkt')

# Configuration and Initialization
aai.settings.api_key = os.environ.get("assemblyai_key")
lemur_endpoint = "https://api.assemblyai.com/lemur/v3/generate/task"
headers = {
    "Authorization": os.environ.get("assemblyai_key")
}
FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"


def get_transcript(transcript_id=None):
    if transcript_id:
        return aai.Transcript.get_by_id(transcript_id)
    else:
        transcriber = aai.Transcriber()
        return transcriber.transcribe(FILE_URL)


def extract_paragraphs_and_sentences(transcript):
    return [p.text for p in transcript.get_paragraphs()], [s.text for s in transcript.get_sentences()]


def sliding_window(sentences, window_size=3):
    return [" ".join(sentences[i:i+window_size]) for i in range(0, len(sentences) - window_size + 1)]


def get_sentence_transformer_embeddings(sentences, model_name):
    model = SentenceTransformer(model_name)
    return model.encode(sentences)


def compare_against_all_granularities(summary_embedding, embeddings_list, k):
    similarities = cosine_similarity([summary_embedding], embeddings_list)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return top_k_indices, similarities[top_k_indices]


def filter_summary_sentences(summary, transcript_id, model_name, k=3):
    # Load the transcript using the provided ID
    transcript = aai.Transcript.get_by_id(transcript_id)
    print(transcript)
    paragraphs = [paragraph.text for paragraph in transcript.get_paragraphs()]
    sentences = [sentence.text for sentence in transcript.get_sentences()]
    model = SentenceTransformer(model_name)
    summary_sentences = nltk.sent_tokenize(summary)
    summary_embeddings = {sentence: model.encode([sentence])[0] for sentence in summary_sentences}
    # Get the embeddings
    sentence_embeddings = model.encode(sentences)
    paragraph_embeddings = model.encode(paragraphs)
    sentence_chunks = sliding_window(sentences)
    chunk_embeddings = model.encode(sentence_chunks)

    
    
    new_summary = ""
    filtered_sentences = []
    sentence_threshold = 0.80
    paragraph_threshold = 0.73
    chunk_threshold = 0.78

    for sentence, embedding in summary_embeddings.items():
        top_k_sentence_indices, top_k_sentence_similarities = compare_against_all_granularities(embedding, sentence_embeddings, k)
        top_k_paragraph_indices, top_k_paragraph_similarities = compare_against_all_granularities(embedding, paragraph_embeddings, k)
        top_k_chunk_indices, top_k_chunk_similarities = compare_against_all_granularities(embedding, chunk_embeddings, k)
        
        if any(sim >= sentence_threshold for sim in top_k_sentence_similarities) or \
           any(sim >= paragraph_threshold for sim in top_k_paragraph_similarities) or \
           any(sim >= chunk_threshold for sim in top_k_chunk_similarities):
            new_summary += sentence + " "
        else:
            filtered_sentences.append(sentence)
    
    return new_summary, filtered_sentences


if __name__ == "__main__":
    # transcript = get_transcript("6mvfr2epvp-65bc-46dc-8b4e-b87487b5da4b")
    # paragraphs, sentences = extract_paragraphs_and_sentences(transcript)
    
    model_name = "infgrad/stella-base-en-v2"
    # sentence_embeddings = get_sentence_transformer_embeddings(sentences, model_name)
    # paragraph_embeddings = get_sentence_transformer_embeddings(paragraphs, model_name)
    # sentence_chunks = sliding_window(sentences)
    # chunk_embeddings = get_sentence_transformer_embeddings(sentence_chunks, model_name)
    
    lemur_summary = "Wildfires in Canada are making the air dirty in many places in the US. Smoke from the fires is traveling through the sky and making it hard to breathe in places like New York and Baltimore. The smoke has tiny pieces in it that can get inside your lungs if you breathe them. This can make you sick, especially kids and older adults. The pieces in the smoke are much more than normal and that's why the air is unhealthy. More people could get sick until the weather changes and moves the smoke away. Fires might happen more often in the future because of climate change, so dirty air could affect more places."
    
    new_summary, filtered_sentences = filter_summary_sentences(lemur_summary, "6vkxdgap5h-8d7b-4559-a702-16bf4c7c3b44", model_name)
    
    print("***************************")
    print("NEW SUMMARY OUTPUT")
    print(new_summary)
    print("***************************")
    print("FILTERED SENTENCES")
    print(filtered_sentences)
