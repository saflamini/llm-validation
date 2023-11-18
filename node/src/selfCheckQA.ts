import fetch from "cross-fetch";
import dotenv from "dotenv";
dotenv.config();
import { AssemblyAI } from 'assemblyai';
import { json } from "stream/consumers";

//New method for applying an LLM check to a QA response
//Take QA output
//Compare to original transcript and ask whether or not it is grounded within the transcript? can you find clear evidence for this answer in the transcript?
//Generate new response based on findings

const API_KEY: string = process.env.assemblyai_api_key || "";

// const SAMPLE_TRANSCRIPT_ID = "6nsz18w71v-b84e-4060-a1e3-072cb2ba281f"
const SAMPLE_TRANSCRIPT_ID = "6nsz0rrkkt-94c9-4bd1-9046-58caa977dadf"

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

//sample json output - this is what we'll be using for the self-check
//this json output has a hallucination - the answer to the first questsion is hallucinated
const originalJSONList = "[{'question': 'What does Obama think about gun rights in America?', 'answer': 'Obama believes that gun ownership has gone too far in America, and that it needs to be reigned in to a substantial degree. He thinks there should be strict regulations like background checks and lengthy approval processes to help prevent guns from getting into more hands'}, {'question': 'What are the primary concerns of the audience members?', 'answer': 'The audience members are concerned that Obama wants to restrict gun rights and limit access to guns, especially for law-abiding citizens. They believe criminals should be held accountable instead.'}, {'question': 'What is the sentiment of this conversation?', 'answer': 'The sentiment is tense but respectful, with Obama trying to reassure gun owners that he supports the Second Amendment while also arguing for targeted regulations to improve public safety. The audience members are skeptical.'}]"
//when we run the below function, we should see that the grounding threshold should not be passed for the first example

const jsonQAList = prepareStringForJson(originalJSONList)

async function main(): Promise<void> {

    console.log("ZERO SHOT QA Response");
    console.log(` ${JSON.parse(jsonQAList).map((qa) => {
        return `${qa.question}: ${qa.answer}`
    }).join("\n")}`)
    console.log("***********************************************************");

    async function selfCheckLeMURQA(jsonQAList: any, transcriptId: string) {

        const client = new AssemblyAI({
            apiKey: API_KEY,
          })
    
        // Parse the JSON string into an array of objects
        const qaList = JSON.parse(jsonQAList);
    
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
            context: "An answer is hallucinated if you cannot find sufficient evidence within the transcript to support it. If there is sufficient evidence, make sure you always put YES, and if there is not sufficient evidence, make sure you always put NO. You should answer NO even if the answer is partially supported by the transcript."
            // final_model: 'basic'
        };
    
        // Call the LeMUR API
        const response = await client.lemur.questionAnswer(lemurRequest)
        
        let qaResults = [];

        const r = response.response

        //label all i
        for (let i = 0; i < r.length; i++) {
            const qaItem = r[i];
            const groundingThresholdPassed = qaItem.answer === 'YES';
            qaResults.push({
                qaItem: qaItem.question,
                groundingThresholdPassed: groundingThresholdPassed,
            });
        }
        return qaResults;
    }

    selfCheckLeMURQA(jsonQAList, SAMPLE_TRANSCRIPT_ID)
    .then(response => console.log(response))
    .catch(error => console.error(error));
}

main();