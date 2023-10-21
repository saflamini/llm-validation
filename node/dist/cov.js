import fetch from "cross-fetch";
const API_KEY = "";
const LEMUR_ENDPOINT = "https://api.assemblyai.com/lemur/v3/generate/task";
const HEADERS = {
    "Authorization": API_KEY
};
const FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3";
const PLACEHOLDER_URL = "https://api.assemblyai-solutions.com/storage/v1/object/public/sam_training_bucket/hello-48300.mp3";
const PLACEHOLDER_ID = "6mg1zsu0yw-0d73-45a2-a631-acc347ec210b"; // REPLACE WITH YOUR OWN PLACEHOLDER ID by transcribing the PLACEHOLDER FILE
const CANADIAN_WILDFIRES_TRANSCRIPT_ID = "6mzwnvtu5k-8724-4bd5-8bd1-62dd2175ed09"; // REPLACE WITH YOUR OWN TRANSCRIPT ID
async function getTranscriptById(id) {
    const response = await fetch(`https://api.assemblyai.com/v2/transcript/${CANADIAN_WILDFIRES_TRANSCRIPT_ID}`, {
        method: 'GET',
        headers: HEADERS
    });
    return await response.json();
}
function splitIntoSentences(text) {
    return text.split('. ');
}
async function lemurTask(prompt, finalModel = "basic", transcriptId) {
    const response = await fetch(LEMUR_ENDPOINT, {
        method: 'POST',
        headers: HEADERS,
        body: JSON.stringify({
            transcript_ids: [transcriptId],
            prompt: prompt,
            final_model: finalModel
        })
    });
    return await response.json();
}
async function main() {
    const transcript = await getTranscriptById(CANADIAN_WILDFIRES_TRANSCRIPT_ID);
    //STEP 1 - summarize the transcript
    const lemurSummary = await lemurTask(`Please summarize this transcript as though I were a 10 year old`, 'basic', CANADIAN_WILDFIRES_TRANSCRIPT_ID);
    const summarySentences = splitIntoSentences(lemurSummary.response);
    console.log("ZERO SHOT SUMMARY");
    console.log(lemurSummary.response);
    console.log("**************************");
    //STEP 2 - generate verification questions. this can be done without inputting all of the original transcript text into context
    //We are side stepping LeMUR's required 'transcript_ids' field here by using a transcript that is only 3 seconds long
    //The content of this 'placeholder transcript' can be found at PLACEHOLDER_URL - it's only text is: 'hello, how are you?'
    const verificationQuestions = await lemurTask(`
        Please identify a series of questions that we can ask to verify the accuracy of this summary. 
        Please go through each sentence of the transcript summary, and generate a question that would help us verify that the summary is accurate:
        ${summarySentences.join("\n")}
    `, 'basic', PLACEHOLDER_ID);
    console.log("VERIFICATION QUESTIONS");
    console.log(verificationQuestions.response);
    console.log("**************************");
    //STEP 3 - answer the verification questions using the transcript
    const answerVerificationQuestions = await lemurTask(`
        Please use the transcript to answer each of the following questions:
        ${verificationQuestions.response}
        Please answer each question with detail where possible. Format your responses as: Question: Answer \n
    `, 'basic', CANADIAN_WILDFIRES_TRANSCRIPT_ID);
    console.log("ANSWER VERIFICATION QUESTIONS");
    console.log(answerVerificationQuestions.response);
    console.log("**************************");
    //STEP 4 - generate a new summary based on the transcript and the verification question answers
    const secondShot = await lemurTask(`
        In addition to the transcript, you have been provided with an initial summary of the transcript as well as a series of questions that can be used to verify the accuracy of the transcript.
        Here is your initial summary:
        ${lemurSummary.response} \n
        Here are the questions you answered about the initial summary:
        ${answerVerificationQuestions.response}
        Please generate a new summary that is fully accurate and entirely based on the transcript and the questions that you have answered about it. If an answer to a question is not in the transcript, you should not include any reference to it in your output.
    `, 'basic', CANADIAN_WILDFIRES_TRANSCRIPT_ID);
    console.log("SECOND SHOT");
    console.log(secondShot.response);
    console.log("**************************");
}
main();
