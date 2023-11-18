import { expect } from 'chai';
import 'mocha';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { AssemblyAI } from 'assemblyai';

import { processLemurQA, computeEmbeddings, splitIntoParagraphs, slidingWindow, selfCheckLeMURQA } from './lemurQAFunctions.js'; // adjust the path as needed

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function prepareStringForJson(input) {
    // First, replace escaped single quotes with a placeholder
    let result = input.replace(/\\'/g, '__SINGLE_QUOTE__');

    // Then, replace single quotes around keys and values with double quotes
    result = result.replace(/([{,]\s*)'([^']+)'\s*:/g, '$1"$2":');
    result = result.replace(/:\s*'([^']+)'/g, ': "$1"');

    // Finally, revert the placeholders back to single quotes
    result = result.replace(/__SINGLE_QUOTE__/g, "'");

    return result;
}

describe('Lemur QA Embeddings Check Testing', function() {
    let successfulPostitiveTestCount = 0;
    let partiallyCorrectTestCount = 0;
    let successfulTestCount = 0;
    this.timeout(30000); // Increase timeout if initialization takes longer

    let pipeline;
    let extractor;

    // Initialize pipeline before tests
    before(async () => {
        const transformers = await import("@xenova/transformers");
        pipeline = transformers.pipeline;
        extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    });

    // Load test data
    const testDataPath = path.join(__dirname, './testset_nov_12_2023.json');
    const testData = JSON.parse(fs.readFileSync(testDataPath, 'utf-8'));

    testData.forEach((record) => {
        it(`should process successful QA results for transcript ID ${record.transcript_id}`, async () => {
            // Prepare transcript data
            const sentences = record.transcript_text.split('. ');
            const paragraphs = splitIntoParagraphs(sentences);
            const chunks = slidingWindow(sentences);

            // Compute embeddings
            const sentenceEmbeddings = await computeEmbeddings(sentences, extractor);
            const paragraphEmbeddings = await computeEmbeddings(paragraphs, extractor);
            const chunkEmbeddings = await computeEmbeddings(chunks, extractor);

            const qaSuccess = prepareStringForJson(record.qa_success);

            console.log(qaSuccess)

            // Process Lemur QA data
            const results = await processLemurQA(JSON.parse(qaSuccess), sentences, paragraphs, chunks, sentenceEmbeddings, paragraphEmbeddings, chunkEmbeddings, extractor);
            console.log(results)
            // ... Evaluate results ...
            let thresholdNotPassedList = []
            for (let i = 0; i < results.length; i++){
                if (results[i].groundingThresholdPassed == false){
                    thresholdNotPassedList.push(i)
                }
            }
            if (thresholdNotPassedList.length == 0){
                successfulPostitiveTestCount++
            }
            expect(thresholdNotPassedList.length).to.equal(0); // assertion
        });
    });

    testData.forEach((record) => {
        it(`should process hallucinated QA results for transcript ID ${record.transcript_id}`, async () => {
            // Prepare transcript data
            const sentences = record.transcript_text.split('. ');
            const paragraphs = splitIntoParagraphs(sentences);
            const chunks = slidingWindow(sentences);

            // Compute embeddings
            const sentenceEmbeddings = await computeEmbeddings(sentences, extractor);
            const paragraphEmbeddings = await computeEmbeddings(paragraphs, extractor);
            const chunkEmbeddings = await computeEmbeddings(chunks, extractor);
            
            const qaHallucinated = JSON.parse(prepareStringForJson(record.qa_hallucinated));

            console.log("QA Result")
            console.log(qaHallucinated)

            const qaHallucinatedLabel = JSON.parse(prepareStringForJson(record.qa_hallucinated_label));
            console.log("Hallucinated QA Result")
            console.log(qaHallucinatedLabel)

            // Process Lemur QA data
            const results = await processLemurQA(qaHallucinated, sentences, paragraphs, chunks, sentenceEmbeddings, paragraphEmbeddings, chunkEmbeddings, extractor);

            let labeledIndices = qaHallucinatedLabel.map(label => 
                qaHallucinated.findIndex(qa => qa.question === label.question)
            );

            console.log("Index of actual hallucinations")
            console.log(labeledIndices)

            // Indices in results where groundingThresholdPassed is false
            let flaggedIndices = results
                .map((result, index) => !result.groundingThresholdPassed ? index : -1)
                .filter(index => index !== -1);

            console.log("Index of predicted hallucinations")
            console.log(flaggedIndices)

            // Check if all labeled hallucinations are correctly flagged
            let partiallyIdentified = labeledIndices.every(index => flaggedIndices.includes(index));
            expect(partiallyIdentified).to.be.true; // Assertion
            if (partiallyIdentified){
                partiallyCorrectTestCount++
            }

            // Check for exact match between labeledIndices and flaggedIndices
            let correctlyIdentified = labeledIndices.length === flaggedIndices.length &&
                                  labeledIndices.every((value, index) => value === flaggedIndices[index]);

            expect(correctlyIdentified).to.be.true; // Assertion
            if (correctlyIdentified){
                successfulTestCount++
            }
        });
    });

    after(() => {
        console.log("Eval of Hallucination Free Responses:")
        console.log("Success rate: ", successfulPostitiveTestCount / testData.length)

        console.log("Eval of Hallucination Responses:")
        console.log("Partial Success Rate (at least one hallucination identified): ", partiallyCorrectTestCount / testData.length)
        console.log(`Total 100% Successful Tests: ${successfulTestCount}`);
    });
});


describe('Lemur LLM Self Check Testing', function() {
    this.timeout(1200000); // Increase timeout if initialization takes longer
    let successfulPostitiveTestCount = 0
    let partiallyCorrectTestCount = 0
    let successfulTestCount = 0
    // Load test data
    const testDataPath = path.join(__dirname, './testset_nov_12_2023.json');
    const testData = JSON.parse(fs.readFileSync(testDataPath, 'utf-8'));

    testData.forEach((record) => {
        it(`should process successful QA results for transcript ID ${record.transcript_id}`, async () => {
            // // Prepare transcript data
            const qaSuccess = prepareStringForJson(record.qa_success);

            // console.log(qaSuccess)

            // Process Lemur QA data
            const results = await selfCheckLeMURQA(qaSuccess, record.transcript_id, AssemblyAI);
            console.log(results)
            // ... Evaluate results ...
            let thresholdNotPassedList = []
            for (let i = 0; i < results.length; i++){
                if (results[i].groundingThresholdPassed == false){
                    thresholdNotPassedList.push(i)
                }
            }
            if (thresholdNotPassedList.length == 0){
                successfulPostitiveTestCount++
            }
            expect(thresholdNotPassedList.length).to.equal(0); // assertion
        });
    });

    for (const record of testData) {
    // testData.forEach(async (record) => {
        it(`should process hallucinated QA results for transcript ID ${record.transcript_id}`, async () => {
            const qaHallucinated = JSON.parse(prepareStringForJson(record.qa_hallucinated));
            const qaHallucinatedLabel = JSON.parse(prepareStringForJson(record.qa_hallucinated_label));
            
            // console.log("QA Result")
            // console.log(qaHallucinated)
            console.log("*************************************************************************************************")
            console.log("Hallucinated QA Result")
            console.log(qaHallucinatedLabel)

            const results = await selfCheckLeMURQA(JSON.stringify(qaHallucinated), record.transcript_id, AssemblyAI);
            
            console.log("Self Check Results")
            console.log(results)

            // Indices in qaHallucinated where hallucinations are labeled
            let labeledIndices = qaHallucinatedLabel.map(label => 
                qaHallucinated.findIndex(qa => qa.question === label.question)
            );
            
            console.log("Index of actual hallucinations")
            console.log(labeledIndices)
    
            // Indices in results where groundingThresholdPassed is false
            let flaggedIndices = results
                .map((result, index) => result.groundingThresholdPassed === false ? index : -1)
                .filter(index => index !== -1);
    
            console.log("Index of predicted hallucinations")
            console.log(flaggedIndices)
            // Check if all labeled hallucinations are correctly flagged
            let partiallyIdentified = labeledIndices.every(index => flaggedIndices.includes(index));
            console.log("At Least Partially Identified: ", partiallyIdentified)
            expect(partiallyIdentified).to.be.true; // Assertion
            if (partiallyIdentified){
                partiallyCorrectTestCount++
            }
            // Check for exact match between labeledIndices and flaggedIndices
            let correctlyIdentified = labeledIndices.length === flaggedIndices.length &&
            labeledIndices.every((value, index) => value === flaggedIndices[index]);
            console.log("100% Correctly Identified: ", correctlyIdentified)
            expect(correctlyIdentified).to.be.true; // Assertion
            if (correctlyIdentified){
                successfulTestCount++
            }
            console.log("*************************************************************************************************")

        });
    };

    after(() => {
        console.log("Eval of Hallucination Free Responses:")
        console.log("Success rate: ", successfulPostitiveTestCount / testData.length)

        console.log("Eval of Hallucination Responses:")
        console.log("Partial Success Rate (at least one hallucination identified): ", partiallyCorrectTestCount / testData.length)
        console.log(`Total 100% Successful Tests: ${successfulTestCount}`);
    });
    
});