import fetch from 'cross-fetch';
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { Tensor2D } from '@tensorflow/tfjs';
import { AssemblyAI } from 'assemblyai';
//using dynamic imports format to make this work w typescript
import("@xenova/transformers").then((module) => {
   let { pipeline } = module;

//replace with your own API key
const API_KEY: string = "";
const AAI_ENDPOINT = "https://api.assemblyai.com/v2/transcript";
const LEMUR_ENDPOINT: string = "https://api.assemblyai.com/lemur/v3/generate/task";
const HEADERS: Record<string, string> = {
    "Authorization": API_KEY
};
const FILE_URL: string = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3";
const PLACEHOLDER_URL = "https://api.assemblyai-solutions.com/storage/v1/object/public/sam_training_bucket/hello-48300.mp3"
const PLACEHOLDER_ID: string = "6mg1zsu0yw-0d73-45a2-a631-acc347ec210b";
const CANADIAN_WILDFIRES_TRANSCRIPT_ID: string = "6mzwnvtu5k-8724-4bd5-8bd1-62dd2175ed09";

interface LemurResponse {
    response: string;
    // Add any other fields from the Lemur API response that you might use.
}

function cosineSimilarity(embeddingA: tf.Tensor, embeddingB: tf.Tensor): number {
    const dotProduct = tf.sum(embeddingA.mul(embeddingB));
    const magA = embeddingA.norm();
    const magB = embeddingB.norm();
    const similarity = dotProduct.div(magA.mul(magB));
    return similarity.arraySync() as number;
}


async function main() {

    function splitIntoSentences(text: string): string[] {
        return text.split('. ');
    }

    async function lemurTask(prompt: string, finalModel: string = "basic", transcriptId: string): Promise<LemurResponse> {
        const response: Response = await fetch(LEMUR_ENDPOINT, {
            method: 'POST',
            headers: HEADERS,
            body: JSON.stringify({
                transcript_ids: [transcriptId],
                prompt: prompt,
                final_model: finalModel
            })
        });
        console.log(response)
        return await response.json() as LemurResponse;
    
    }
    
    const client = new AssemblyAI({
        apiKey: API_KEY
    })

    const config = {
        audio_url: FILE_URL,
    }

    // const transcript = await client.transcripts.create(config);
    const transcript = await client.transcripts.get("6mmrg2wf5b-eb91-4adf-9392-8e05bcaabefc");
    console.log(transcript.id);

    const lemurSummary = await client.lemur.summary({
        transcript_ids: [transcript.id],
        context: "Please summarize this transcript as though I were a 10 year old",
        final_model: "basic"
    })


    const summarySentences: string[] = splitIntoSentences(lemurSummary.response);


    console.log("ZERO SHOT SUMMARY");
    console.log(lemurSummary.response);
    console.log("**************************");

    //STEP 2 - generate verification questions. this can be done without inputting all of the original transcript text into context
    //We are side stepping LeMUR's required 'transcript_ids' field here by using a transcript that is only 3 seconds long
    //The content of this 'placeholder transcript' can be found at PLACEHOLDER_URL - it's only text is: 'hello, how are you?'
    const verificationQuestions: LemurResponse = await lemurTask(`
            Please identify a series of questions that we can ask to verify the accuracy of this summary. 
            Please go through each sentence of the transcript summary, and generate a question that would help us verify that the summary is accurate:
            ${summarySentences.join("\n")}
        `, 'basic', PLACEHOLDER_ID);

    console.log("VERIFICATION QUESTIONS");
    console.log(verificationQuestions.response);
    console.log("**************************");

    //STEP 3 - answer the verification questions using the transcript
    const answerVerificationQuestions: LemurResponse = await lemurTask(`
        Please use the transcript to answer each of the following questions:
        ${verificationQuestions.response}
        Please answer each question with detail where possible. Format your responses as: Question: Answer \n
        `, 'basic', CANADIAN_WILDFIRES_TRANSCRIPT_ID);

    console.log("ANSWER VERIFICATION QUESTIONS");
    console.log(answerVerificationQuestions.response);
    console.log("**************************");

    //STEP 4 - generate a new summary based on the transcript and the verification question answers
    const secondShot: LemurResponse = await lemurTask(`
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

    //STEP 2 - VERIFICATION W EMBEDDING CITATIONS
    //why all-MiniLM-L6-v2? https://huggingface.co/Xenova/all-MiniLM-L6-v2
    //this is the model that we are using to compute the embeddings
    //for some reason it's labeled as a 'feature-extraction' pipeline, but this is the same model used in our python instance

    const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    
    const transcriptSentences: string[] = [];
    let transcriptSentenceResponse = await client.transcripts.sentences(transcript.id)
    for (let i = 0; i < transcriptSentenceResponse.sentences.length; i++) {
        transcriptSentences.push(transcriptSentenceResponse.sentences[i].text)
    }

    async function computeEmbeddings(sentences: string[]): Promise<tf.Tensor[]> {
        // let embeddings: any[] = [];
        let embeddings: tf.Tensor[] = [];
        let embedding = await extractor(sentences, { pooling: 'mean', normalize: true });
        for (let i = 0; i < sentences.length; i++) {
            let embedding = await extractor(sentences[i], { pooling: 'mean', normalize: true });           
            let tensor = tf.tensor(embedding.data);
            embeddings.push(tensor);
        }
        
        return embeddings
    }
    
    const finalSummarySentences = splitIntoSentences(secondShot.response)

    const transcriptSentenceEmbeddings = await computeEmbeddings(transcriptSentences);
    const summarySentenceEmbeddings = await computeEmbeddings(finalSummarySentences);

    const threshold = 0.6;

    // This will store the new summary sentences that meet the threshold
    let newSummarySentences: string[] = [];

    // This will store the sentences that don't meet the threshold
    let filteredSentences: string[] = [];

    finalSummarySentences.forEach((finalSummarySentence, i) => {
        let maxSimilarity = -Infinity;  // Initialize to a very low value
        let mostSimilarTranscriptSentence = "";

        transcriptSentences.forEach((transcriptSentence, j) => {
            let similarity = cosineSimilarity(summarySentenceEmbeddings[i], transcriptSentenceEmbeddings[j]);
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                mostSimilarTranscriptSentence = transcriptSentence;
            }
        });

        // Print the most similar sentence and its similarity score
        console.log(`Summary Sentence: ${finalSummarySentence}`);
        console.log(`Most Similar Transcript Sentence: ${mostSimilarTranscriptSentence}`);
        console.log(`Similarity Score: ${maxSimilarity.toFixed(2)}`);
        console.log('-----------------------------------');

        if (maxSimilarity >= threshold) {
            newSummarySentences.push(finalSummarySentence);
        } else {
            filteredSentences.push(finalSummarySentence);
        }
    });

    console.log("NEW SUMMARY OUTPUT:");
    console.log(newSummarySentences.join(". "));
    console.log("\nFILTERED SENTENCES:");
    console.log(filteredSentences.join(". "));
}

main();
});