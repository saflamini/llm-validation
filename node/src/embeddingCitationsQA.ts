import fetch from 'cross-fetch';
import * as tf from '@tensorflow/tfjs';
import { AssemblyAI } from 'assemblyai';
import dotenv from "dotenv";
dotenv.config();

// Dynamic import for the transformers model
import("@xenova/transformers").then(async (module) => {
    let { pipeline } = module;

    const API_KEY: string = process.env.assemblyai_api_key || "";
    const HEADERS = {
        "Authorization": "Bearer " + API_KEY
    };

    const FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3";

    async function getTranscriptById(id: string) {
        const response = await fetch(`https://api.assemblyai.com/v2/transcript/${id}`, {
            method: 'GET',
            headers: HEADERS
        });
        return await response.json();
    }

    async function computeEmbeddings(sentences: string[]): Promise<tf.Tensor[]> {
        let embeddings: tf.Tensor[] = [];
        const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        for (let sentence of sentences) {
            let embedding = await extractor(sentence, { pooling: 'mean', normalize: true });
            let tensor = tf.tensor(embedding.data);
            embeddings.push(tensor);
        }
        return embeddings;
    }

    function splitIntoParagraphs(sentences: string[]) {
        return sentences.reduce((acc, sentence, idx) => {
            const paragraphIdx = Math.floor(idx / 5);
            acc[paragraphIdx] = (acc[paragraphIdx] || '') + sentence + ' ';
            return acc;
        }, [] as string[]);
    }

    function slidingWindow(sentences: string[], windowSize: number = 3) {
        return sentences.map((_, idx, arr) => arr.slice(idx, idx + windowSize).join(' ')).slice(0, -windowSize + 1);
    }

    // Define thresholds for different granularities
const GRANULARITY_THRESHOLDS = {
    "sentence": 0.75,
    "paragraph": 0.69,
    "chunk": 0.72
};

function cosineSimilarity(embeddingA: tf.Tensor, embeddingB: tf.Tensor): number {
    const dotProduct = tf.sum(embeddingA.mul(embeddingB));
    const magA = embeddingA.norm();
    const magB = embeddingB.norm();
    const similarity = dotProduct.div(magA.mul(magB)).dataSync()[0];
    return similarity as number; // Ensure it's a number
}

function findTopSimilarities(qaEmbedding: tf.Tensor, embeddings: tf.Tensor[], k: number = 3): { indices: number[], similarities: number[] } {
    const similarities = embeddings.map(embedding => cosineSimilarity(qaEmbedding, embedding));
    const sortedIndices = similarities.map((_, idx) => idx).sort((a, b) => similarities[b] - similarities[a]);
    return {
        indices: sortedIndices.slice(0, k),
        similarities: sortedIndices.slice(0, k).map(idx => similarities[idx])
    };
}

async function processLemurQA(lemurQA: any[], sentences: string[], paragraphs: string[], chunks: string[], sentenceEmbeddings: tf.Tensor[], paragraphEmbeddings: tf.Tensor[], chunkEmbeddings: tf.Tensor[]) {
    const results = [];

    for (const qaItem of lemurQA) {
        const qaEmbedding = await computeEmbeddings([qaItem.answer]);

        const { indices: topKSentenceIndices, similarities: topKSentenceSimilarities } = findTopSimilarities(qaEmbedding[0], sentenceEmbeddings);
        const { indices: topKParagraphIndices, similarities: topKParagraphSimilarities } = findTopSimilarities(qaEmbedding[0], paragraphEmbeddings);
        const { indices: topKChunkIndices, similarities: topKChunkSimilarities } = findTopSimilarities(qaEmbedding[0], chunkEmbeddings);

        const sentenceMargin = topKSentenceSimilarities[0] - GRANULARITY_THRESHOLDS["sentence"];
        const paragraphMargin = topKParagraphSimilarities[0] - GRANULARITY_THRESHOLDS["paragraph"];
        const chunkMargin = topKChunkSimilarities[0] - GRANULARITY_THRESHOLDS["chunk"];

        let transcriptRef: string, mostSimilarGranularity: number;
        if (sentenceMargin > paragraphMargin && sentenceMargin > chunkMargin && sentenceMargin > 0) {
            transcriptRef = sentences[topKSentenceIndices[0]];
            mostSimilarGranularity = topKSentenceSimilarities[0];
        } else if (paragraphMargin > chunkMargin && paragraphMargin > 0) {
            transcriptRef = paragraphs[topKParagraphIndices[0]];
            mostSimilarGranularity = topKParagraphSimilarities[0];
        } else if (chunkMargin > 0) {
            transcriptRef = chunks[topKChunkIndices[0]];
            mostSimilarGranularity = topKChunkSimilarities[0];
        } else {
            transcriptRef = "No reference met the threshold.";
            mostSimilarGranularity = topKSentenceSimilarities[0];
        }

        const groundingThresholdPassed = Math.max(sentenceMargin, paragraphMargin, chunkMargin) > 0;

        results.push({
            question: qaItem.question,
            answer: qaItem.answer,
            citation: {
                reference: transcriptRef,
                similarityScore: mostSimilarGranularity
            },
            groundingThresholdPassed
        });
    }

    return results;
}

async function main() {
    // Initialize AssemblyAI client
    const client = new AssemblyAI({
        apiKey: API_KEY
    });

    // Fetch transcript
    const transcriptId = "6nsz0rrkkt-94c9-4bd1-9046-58caa977dadf"; // Replace with actual transcript ID
    const transcript = await getTranscriptById(transcriptId);

    // Process transcript and compute embeddings
    const sentences = transcript.text.split('. ');
    const paragraphs = splitIntoParagraphs(sentences);
    const chunks = slidingWindow(sentences);
    const sentenceEmbeddings = await computeEmbeddings(sentences);
    const paragraphEmbeddings = await computeEmbeddings(paragraphs);
    const chunkEmbeddings = await computeEmbeddings(chunks);

    // Define your Lemur QA data
    const lemurQAOutput = [{"question": "What locations in california are the interviewees from?", "answer": "The interviewees are from various locations in California including Ventura County, San Diego, Los Angeles, and Sonoma County."}, {"question": "What are the main reasons why people are moving to Texas from California?", "answer": "The main reasons people are moving from California to Texas are the high cost of living, high housing prices, high taxes, overregulation of businesses, increased crime, homelessness, and liberal politics in California."}, {"question": "What demographics are moving out of California in the highest quantities?", "answer": "The transcript does not cite specific demographic data, but interviewees mentioned that California is becoming less friendly to young families in comparison to older, affluent people."}, {"question": "What do the people who moved to Texas think of Texas?", "answer": "Most of the people who moved from California to Texas are very happy with their decision. They cite the lower cost of living, bigger and more affordable houses, feeling of safety, and being around more politically like-minded people as the main benefits."}, {"question": "Why is California so much more expensive than Texas?", "answer": "California is more expensive due to very high housing costs, higher taxes, more regulations that increase business costs, and an overall higher cost of goods and services."}, {"question": "How much more expensive is California than the rest of the country?", "answer": "California\'s cost of living is about 15% higher than the overall United States."}];


    // Process Lemur QA data
    const results = await processLemurQA(lemurQAOutput, sentences, paragraphs, chunks, sentenceEmbeddings, paragraphEmbeddings, chunkEmbeddings);

    console.log(results);
}

main().catch(error => {
    console.error("An error occurred:", error);
});
    
})