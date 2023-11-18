import fetch from 'cross-fetch';
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { Tensor2D } from '@tensorflow/tfjs';
import { log } from 'console';
import { AssemblyAI } from 'assemblyai';
import http from 'http';
import querystring from 'querystring';
import url from 'url';
import dotenv from "dotenv";
dotenv.config();
//using dynamic imports format to make this work w typescript
import("@xenova/transformers").then((module) => {
   let { pipeline } = module;

//replace with your own API key
const API_KEY: string = process.env.assemblyai_api_key || "";

// const LEMUR_ENDPOINT = "https://api.assemblyai.com/lemur/v3/generate/task";
const LEMUR_ENDPOINT = "https://api.staging.assemblyai-labs.com/lemur/v3/generate/" // this is the staging endpoint, using for testing
const AAI_ENDPOINT = "https://api.assemblyai.com/v2/transcript";
const HEADERS = {
    "Authorization": "Bearer" + API_KEY
};

//this is a sample file from the assembly docs. can replace as needed
const FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3";

//function to get transcript if it's needed
async function getTranscriptById(id: string) {
    const response = await fetch(`${AAI_ENDPOINT}/${id}`, {
        method: 'GET',
        headers: HEADERS
    });
    return await response.json();
}

function cosineSimilarity(embeddingA: tf.Tensor, embeddingB: tf.Tensor): number {
    const dotProduct = tf.sum(embeddingA.mul(embeddingB));
    const magA = embeddingA.norm();
    const magB = embeddingB.norm();
    const similarity = dotProduct.div(magA.mul(magB));
    return similarity.arraySync() as number;
}


async function main() {

    //why all-MiniLM-L6-v2? https://huggingface.co/Xenova/all-MiniLM-L6-v2
    //this is the model that we are using to compute the embeddings
    //for some reason it's labeled as a 'feature-extraction' pipeline, but this is the same model used in our python instance

    const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    
    const client = new AssemblyAI({
        apiKey: API_KEY
    })

    const config = {
        audio_url: FILE_URL,
    }

    // const transcript = await client.transcripts.create(config);
    const transcript = await client.transcripts.get("6mmrg2wf5b-eb91-4adf-9392-8e05bcaabefc");
    console.log(transcript.id);
    //get the sentences from the transcript response
    const transcriptSentences: string[] = [];
    let transcriptSentenceResponse = await client.transcripts.sentences(transcript.id)
    for (let i = 0; i < transcriptSentenceResponse.sentences.length; i++) {
        transcriptSentences.push(transcriptSentenceResponse.sentences[i].text)
    }

    async function computeEmbeddings(sentences: string[]): Promise<tf.Tensor[]> {
        let embeddings: tf.Tensor[] = [];
        let embedding = await extractor(sentences, { pooling: 'mean', normalize: true });
        for (let i = 0; i < sentences.length; i++) {
            let embedding = await extractor(sentences[i], { pooling: 'mean', normalize: true });
            let tensor = tf.tensor(embedding.data);
            embeddings.push(tensor);
        }
        
        return embeddings
    }
    
    //logic to create a new lemur summary is here
    // const summaryResponse = await client.lemur.summary({
    //     transcript_ids: [transcript.id],
    //     context: "Please summarize this transcript as though I were a 10 year old",
    //     final_model: "basic"
    // })

    const storedSummaryResponse = "Wildfires in Canada are making the air dirty in many places in the US. Smoke from the fires is traveling through the sky and making it hard to breathe in places like New York and Baltimore. The smoke has tiny pieces in it that can get inside your lungs if you breathe them. This can make you sick, especially kids and older adults. The pieces in the smoke are much more than normal and that's why the air is unhealthy. More people could get sick until the weather changes and moves the smoke away. Fires might happen more often in the future because of climate change, so dirty air could affect more places."

    const summarySentences = storedSummaryResponse.split(". ");

    const transcriptSentenceEmbeddings = await computeEmbeddings(transcriptSentences);
    const summarySentenceEmbeddings = await computeEmbeddings(summarySentences);

    const threshold = 0.6;

    // This will store the new summary sentences that meet the threshold
    let newSummarySentences: string[] = [];

    // This will store the sentences that don't meet the threshold
    let filteredSentences: string[] = [];

    summarySentences.forEach((summarySentence, i) => {
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
        console.log(`Summary Sentence: ${summarySentence}`);
        console.log(`Most Similar Transcript Sentence: ${mostSimilarTranscriptSentence}`);
        console.log(`Similarity Score: ${maxSimilarity.toFixed(2)}`);
        console.log('-----------------------------------');

        if (maxSimilarity >= threshold) {
            newSummarySentences.push(summarySentence);
        } else {
            filteredSentences.push(summarySentence);
        }
    });

    console.log("NEW SUMMARY OUTPUT:");
    console.log(newSummarySentences.join(". "));
    console.log("\nFILTERED SENTENCES:");
    console.log(filteredSentences.join(". "));
}

main();
});