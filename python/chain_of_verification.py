from pydantic import BaseModel
import assemblyai as aai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import requests

# Replace with your API token
aai.settings.api_key = f"KEY"
lemur_endpoint = "https://api.assemblyai.com/lemur/v3/generate/task"
headers = {
    "Authorization": "KEY"
}

# URL of the file to transcribe
PLACEHOLDER_URL = "https://api.assemblyai-solutions.com/storage/v1/object/public/sam_training_bucket/hello-48300.mp3"
FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

placeholder_id = "6mg1zsu0yw-0d73-45a2-a631-acc347ec210b" #empty file (hello, how are you...)
canadian_wildfires_transcript = "6mzwnvtu5k-8724-4bd5-8bd1-62dd2175ed09" #id of the transcribed file url
# transcriber = aai.Transcriber()
# transcript = transcriber.transcribe(FILE_URL)
transcript = aai.Transcript.get_by_id(canadian_wildfires_transcript)
placeholder_transcript = aai.Transcript.get_by_id(placeholder_id)

lemur_summary = transcript.lemur.summarize(
    context="Please summarize this transcript as though I were a 10 year old",
    answer_format="paragaraphs",
    final_model="basic"
)

# Split the summary into sentences
# summary_sentences = lemur_summary.response.split(". ")
summary_sentences = nltk.sent_tokenize(lemur_summary.response)

print("ZERO SHOT SUMMARY")
print(lemur_summary.response)
print("**************************")

#we get better results when using the underlying transcript itself, but this is still not bad using the placeholder and it saves $$$
verification_questions = placeholder_transcript.lemur.task(
    prompt=f"""
    Please identify a series of questions that we can ask to verify the accuracy of this summary. You should focus entirely on the content of the summary, not the style or grammar.
    If a claim is made within the summary, you should generate a question that would help us verify that the claim came directly from underlying source material. For example:

    For example, if in your summary you state that 'Barack Obama is from Utah,' we should generate a verification question that asks 'Where is Barack Obama from?'

    These questions should scrutinize the summary so that we can be confident that it is accurate.

    Please go through each sentence of the transcript summary, and generate a question that would help us verify that the summary is accurate:

    {summary_sentences}
    """,
    final_model="basic"
)

# verification_questions = transcript.lemur.task(
#     prompt=f"""
#     Please identify a series of questions that we can ask to verify the accuracy of this summary. You should focus entirely on the content of the summary, not the style or grammar.
#     If a claim is made within the summary, you should generate a question that would help us verify that the claim came directly from underlying source material. For example:

#     For example, if in your summary you state that 'Barack Obama is from Utah,' we should generate a verification question that asks 'Where is Barack Obama from?'

#     These questions should scrutinize the summary so that we can be confident that it is accurate.

#     Please go through each sentence of the transcript summary, and generate a question that would help us verify that the summary is accurate:

#     {summary_sentences}
#     """,
#     final_model="basic"
# )

print("VERIFICATION QUESTIONS")
print(verification_questions.response)
print("**************************")

#it may be worth using default model to answer these questions
answer_verifcation_questions = transcript.lemur.task(
    prompt=f"""
    Please use the transcript to answer each of the following questions:

    {verification_questions.response}

    Please answer each question with detail where possible. Format your responses as: Question: Answer \n
    """,
    final_model="basic"
)

print("ANSWER VERIFICATION QUESTIONS")
print(answer_verifcation_questions.response)
print("**************************")

second_shot = transcript.lemur.task(
    prompt=f"""
    In addition to the transcript, you have been provided with an initial summary of the transcript as well as a series of questions that can be used to verify the accuracy of the transcript.

    Here is your initial summary:

    {lemur_summary.response} \n

    Here are the questions you answered about the initial summary:

    {answer_verifcation_questions.response}

    Please generate a new summary that is fully accurate based on the transcript and the questions that you have answered about it.
    """,
    final_model="basic"
)

print("SECOND SHOT")
print(second_shot.response)
print("**************************")