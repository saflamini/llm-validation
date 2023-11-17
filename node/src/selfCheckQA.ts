import fetch from "cross-fetch";
import dotenv from "dotenv";
dotenv.config();
import { AssemblyAI } from 'assemblyai';

//New method for applying an LLM check to a QA response
//Take QA output
//Compare to original transcript and ask whether or not it is grounded within the transcript? can you find clear evidence for this answer in the transcript?
//Generate new response based on findings

const API_KEY: string = process.env.assemblyai_api_key || "";
const LEMUR_ENDPOINT: string = "https://api.assemblyai.com/lemur/v3/generate/task";
const HEADERS: Record<string, string> = {
    "Authorization": API_KEY
};
const FILE_URL: string = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3";
const PLACEHOLDER_URL = "https://api.assemblyai-solutions.com/storage/v1/object/public/sam_training_bucket/hello-48300.mp3"
const PLACEHOLDER_ID: string = "6mg1zsu0yw-0d73-45a2-a631-acc347ec210b"; // REPLACE WITH YOUR OWN PLACEHOLDER ID by transcribing the PLACEHOLDER FILE
const CANADIAN_WILDFIRES_TRANSCRIPT_ID: string = "6mzwnvtu5k-8724-4bd5-8bd1-62dd2175ed09"; // REPLACE WITH YOUR OWN TRANSCRIPT ID
// const SAMPLE_TRANSCRIPT_ID = "6nsz18w71v-b84e-4060-a1e3-072cb2ba281f"
const SAMPLE_TRANSCRIPT_ID = "6nsz0rrkkt-94c9-4bd1-9046-58caa977dadf"

interface LemurResponse {
    response: string;
    // Add any other fields from the Lemur API response that you might use.
}

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

async function getTranscriptById(id: string): Promise<LemurResponse> {
    const response: Response = await fetch(`https://api.assemblyai.com/v2/transcript/${CANADIAN_WILDFIRES_TRANSCRIPT_ID}`, {
        method: 'GET',
        headers: HEADERS
    });
    return await response.json() as LemurResponse;
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
    return await response.json() as LemurResponse;
}

const originalJSONList = "[{'question': 'What locations in california are the interviewees from?', 'answer': 'The interviewees are from various locations in California including Ventura County, San Diego, Los Angeles, and Sonoma County.'}, {'question': 'What are the main reasons why people are moving to Texas from California?', 'answer': 'The main reasons people are moving from California to Texas are the high cost of living, high housing prices, high taxes, overregulation of businesses, increased crime, homelessness, and liberal politics in California.'}, {'question': 'What demographics are moving out of California in the highest quantities?', 'answer': 'The transcript does not cite specific demographic data, but interviees mentioned that California is becoming less friendly to young families in comparison to older, affluent people.'}, {'question': 'What do the people who moved to Texas think of Texas?', 'answer': 'Most of the people who moved from California to Texas are very happy with their decision. They cite the lower cost of living, bigger and more affordable houses, feeling of safety, and being around more politically like-minded people as the main benefits.'}, {'question': 'Why is California so much more expensive than Texas?', 'answer': 'California is more expensive due to very high housing costs, higher taxes, more regulations that increase business costs, and an overall higher cost of goods and services.'}, {'question': 'How much more expensive is California than the rest of the country?', 'answer': \"California's cost of living is about 15% higher than the overall United States.\"}]"

const jsonQAList = prepareStringForJson(originalJSONList)


async function main(): Promise<void> {

    console.log("ZERO SHOT QA Response");
    console.log(` ${jsonQAList.map((qa) => {
        return `${qa.question}: ${qa.answer}`
    }).join("\n")}`)
    // console.log(lemurSummary.response);
    console.log("**************************");

    //STEP 2 - generate verification questions. this can be done without inputting all of the original transcript text into context
    //We are side stepping LeMUR's required 'transcript_ids' field here by using a transcript that is only 3 seconds long
    //The content of this 'placeholder transcript' can be found at PLACEHOLDER_URL - it's only text is: 'hello, how are you?'
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