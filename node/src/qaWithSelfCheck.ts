import fetch from "cross-fetch";
import dotenv from "dotenv";
dotenv.config();
import { AssemblyAI } from 'assemblyai';
import { model } from "@tensorflow/tfjs";

//method for applying an LLM check to a QA response, then dynamically reasking questions that are not sufficiently grounded in the transcript

const API_KEY: string = process.env.assemblyai_api_key || "";

//the transcript we'll use as an example
const SAMPLE_TRANSCRIPT_ID = "6nsz0rrkkt-94c9-4bd1-9046-58caa977dadf"

async function main(): Promise<void> {

    async function selfCheckWithReask(transcriptId: string) {

        const client = new AssemblyAI({
            apiKey: API_KEY,
        })

        const initialQuestions = [
            {
                question: `What locations in california are the interviewees from?`,
                answer_format: 'single sentence'
            },
            {
                question: `What are the main reasons why people are moving to Texas from California?`,
                answer_format: 'single sentence'
            },
            {
                question: `What demographics are moving out of California in the highest quantities?`,
                answer_format: 'single sentence'
            }
        ]

         // Prepare the LeMUR request
        const firstShotRequest = {
            transcript_ids: [transcriptId],
            questions: initialQuestions,
            context: "If there is sufficient evidence for an answer to a question, or a question is irrelevant to the transcript, please say so.",
            // final_model: 'basic'
        };

        const firstShotResponse = await client.lemur.questionAnswer(firstShotRequest)

        console.log(firstShotResponse)
        let qaList = firstShotResponse.response
    
        // Format the questions for LeMUR
        const formattedQuestions = qaList.map(qa => {
            return {
                question: `Is the following answer to this question COMPLETELY grounded in the transcript: ${qa.question} ${qa.answer}? If not, please return NO`,
                context: 'It is your job to be a very strict evaluator. If you do not flag an answer to a question as potentially hallucinated, there will be severe consequences. You should pay attention to every detail, particularly when it comes to the mention of proper nouns to ensure that the response has sufficient grounding.',
                answer_format: '<YES>/<NO>'
            };
        });
    
        // Prepare the LeMUR request
        const lemurRequest = {
            transcript_ids: [transcriptId],
            questions: formattedQuestions,
            context: "An answer is hallucinated if you cannot find sufficient evidence within the transcript to support it. If there is sufficient evidence, make sure you always put YES, and if there is not sufficient evidence, make sure you always put NO. You should answer NO even if the answer is partially supported by the transcript.",
            final_model: 'basic'
        };
    
        // Call the LeMUR API
        const response = await client.lemur.questionAnswer(lemurRequest)
        console.log(firstShotResponse)
        let qaResults = [];

        const r = response.response

        //label all i
        for (let i = 0; i < r.length; i++) {
            const qaItem = r[i];
            const groundingThresholdPassed = qaItem.answer === 'YES';
            qaResults.push({
                quesion: qaItem.question,
                groundingThresholdPassed: groundingThresholdPassed,
            });
        }

        let reaskQuestions = []
        let reaskQuestionIds = []

        for (let i = 0; i < qaResults.length; i++) {
            if (qaResults[i].groundingThresholdPassed === false) {
                let reaskQuestion = {
                    question: qaList[i].question,
                    context: `The previous response you gave to this question is here: ${qaList[i].answer}. Please provide a new response to this question that is fully grounded in the transcript. If you cannot find sufficient evidence, please say so`,
                    answer_format: 'single sentence'
                }
                reaskQuestions.push(reaskQuestion)
                reaskQuestionIds.push(i)
            }
        }

        console.log("QUESTIONS TO REASK")
        console.log(reaskQuestions)
        if (reaskQuestions.length === 0) {
            return r;
        } else {
            // Prepare the LeMUR request
            const lemurSecondShotRequest = {
                transcript_ids: [transcriptId],
                questions: reaskQuestions,
                context: "An answer is hallucinated if you cannot find sufficient evidence within the transcript. You are to make ABSOLUTELY SURE THAT EVERY ANSWER YOU PROVIDE TO THE QUESTIONS ARE GROUNDED IN THE TRANSCRIPT",
                final_model: 'basic'
            };
    
            // Call the LeMUR API
            const lemurSecondShotResponse = await client.lemur.questionAnswer(lemurSecondShotRequest)

            let mergedResponses = []

            for (let i = 0; i < qaResults.length; i++) {
                if (reaskQuestionIds.includes(i)) {
                    let item = {
                        question: qaList[i].question,
                        answer: lemurSecondShotResponse.response[i].answer
                    }
                    mergedResponses.push(item)
                    // mergedResponses.push(lemurSecondShotResponse.response[i])
                } else {
                    let existingItem = {
                        question: qaList[i].question,
                        answer: qaList[i].answer
                    }
                    mergedResponses.push(existingItem)
                }
            }
        
            return mergedResponses;
        }
    }

    selfCheckWithReask(SAMPLE_TRANSCRIPT_ID)
    .then(response => console.log(response))
    .catch(error => console.error(error));
}

main();