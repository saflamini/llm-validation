### How to Use This Section

This section is structured as follows:

1) /summary contains a few examples showing how chain of verification and embeddings can be used to detect hallucinations within transcripts

2) The other files regarding QA do the following

`embeddingCitationsQA.ts` - this shows how to use embeddings to identify potential hallucinations in QA responses from LeMUR

`selfCheckQA.ts` - this shows how to use LeMUR itself to identify potential hallucinations in QA responses from LeMUR

`qaWithSelfCheck.ts` - this shows how to use selfCheck with an LLM to generate a QA response and automatically evaluate the response 


`lemurQAFunctions.ts` - this is a file that contains helper functions we'll use within our /test folder

testQA.ts evaluates the effectiveness of the embeddings citations and self check with LLM methods on catching hallucinations. testset_nov_12_2023.json contains a small (25 record) dataset which includes successful QA responses & labeled hallucinated QA responses. This is a challenging dataset. The 'hallucinations' within the hallucinated responses have been deliberately curated to be plausible sounding based on the transcript.

#### Findings from testing:
the embeddingCitations model is not very effective. There are many false positives and most of the actual hallucinations are either only partially identified or completely missed.

the method found in selfCheckQA and qaWithSelf check (using an LLM to detect a hallucination within the output) performed much better. There were few false positives, and a > 50% success rate in identifying hallucinations.

The areas where the LLM delivered false positives in its check occured when the question required a more 'high level' evaluation of the transcript. Questions such as 'what was the sentiment of this conversation' are flagged as ungrounded because there are no direct citations which are generated in support of such an observation.


#### Test results:
Eval of Hallucination Free Responses: 72%
Eval of Hallucination Responses:
Partial Success Rate (at least one hallucination identified): 64%
Total 100% Successful Tests: 56%