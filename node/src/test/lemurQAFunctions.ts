// lemurQAProcessor.ts
import fetch from 'cross-fetch';
import * as tf from '@tensorflow/tfjs';
import { AssemblyAI } from 'assemblyai';
import dotenv from "dotenv";
dotenv.config();
// Import other necessary dependencies and utilities


export async function computeEmbeddings(sentences: string[], extractor): Promise<tf.Tensor[]> {
    let embeddings: tf.Tensor[] = [];
    for (let sentence of sentences) {
        let embedding = await extractor(sentence, { pooling: 'mean', normalize: true });
        let tensor = tf.tensor(embedding.data);
        embeddings.push(tensor);
    }
    return embeddings;
}

export function splitIntoParagraphs(sentences: string[]) {
    return sentences.reduce((acc, sentence, idx) => {
        const paragraphIdx = Math.floor(idx / 5);
        acc[paragraphIdx] = (acc[paragraphIdx] || '') + sentence + ' ';
        return acc;
    }, [] as string[]);
}

export function slidingWindow(sentences: string[], windowSize: number = 3) {
    return sentences.map((_, idx, arr) => arr.slice(idx, idx + windowSize).join(' ')).slice(0, -windowSize + 1);
}

export function cosineSimilarity(embeddingA: tf.Tensor, embeddingB: tf.Tensor): number {
    const dotProduct = tf.sum(embeddingA.mul(embeddingB));
    const magA = embeddingA.norm();
    const magB = embeddingB.norm();
    const similarity = dotProduct.div(magA.mul(magB)).dataSync()[0];
    return similarity as number; // Ensure it's a number
}

export function findTopSimilarities(qaEmbedding: tf.Tensor, embeddings: tf.Tensor[], k: number = 3): { indices: number[], similarities: number[] } {
    const similarities = embeddings.map(embedding => cosineSimilarity(qaEmbedding, embedding));
    const sortedIndices = similarities.map((_, idx) => idx).sort((a, b) => similarities[b] - similarities[a]);
    return {
        indices: sortedIndices.slice(0, k),
        similarities: sortedIndices.slice(0, k).map(idx => similarities[idx])
    };
}

// Define thresholds for different granularities
export const GRANULARITY_THRESHOLDS = {
    "sentence": 0.50,
    "paragraph": 0.50,
    "chunk": 0.50
};

export async function processLemurQA(lemurQA: any[], sentences: string[], paragraphs: string[], chunks: string[], sentenceEmbeddings: tf.Tensor[], paragraphEmbeddings: tf.Tensor[], chunkEmbeddings: tf.Tensor[], extractor) {
    const results = [];

    for (const qaItem of lemurQA) {
        const qaEmbedding = await computeEmbeddings([qaItem.answer], extractor);

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


export async function selfCheckLeMURQA(jsonQAList: any, transcriptId: string, assemblyai: any) {
    try {

        const client = new assemblyai({
            apiKey: process.env.assemblyai_api_key,
        })

        // Parse the JSON string into an array of objects
        const qaList = JSON.parse(jsonQAList);


        // Format the questions for LeMUR
        const formattedQuestions = qaList.map(qa => ({
            question: `Is the following answer to this question COMPLETELY grounded in the transcript: <question-answer> ${qa.question} ${qa.answer}? <question-answer> If not, please return NO`,
            context: 'It is your job to be a very strict evaluator. If you do not flag an answer to a question as potentially hallucinated, there will be severe consequences. You should pay attention to every detail, particularly when it comes to the mention of proper nouns to ensure that the response has sufficient grounding.',
            answer_format: '<YES>/<NO>'
        }));

        // Prepare the LeMUR request
        const lemurRequest = {
            transcript_ids: [transcriptId],
            questions: formattedQuestions,
            context: "An answer is hallucinated if you cannot find sufficient evidence within the transcript to support it. If there is sufficient evidence, make sure you always put YES, and if there is not sufficient evidence, make sure you always put NO. You should answer NO even if the answer is partially supported by the transcript.",
            model: 'basic'
        };

        // Call the LeMUR API
        let response: any;
        try {
            response = await client.lemur.questionAnswer(lemurRequest);

        } catch (error) {
            console.error('Error in selfCheckLeMURQA:', error);
        }

        // console.log(response)

        let qaResults = [];

        for (const qaItem of response.response) {
            qaResults.push({
                qaItem: qaItem.question,
                groundingThresholdPassed: qaItem.answer === 'YES',
            });
        }

        return qaResults;
    } catch (error) {
        console.error('Error in selfCheckLeMURQA:', error);
        throw error;
    }
}
