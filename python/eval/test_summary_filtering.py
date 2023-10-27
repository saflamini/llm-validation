import pytest
import sys
import assemblyai as aai
import os
sys.path.append('/Users/sam_flamini/devPortfolio/llm-validation/python/')
from summary_embeddings_citations import filter_summary_sentences

aai.settings.api_key = os.environ.get("assemblyai_key")

#need to fill this with approx 30 examples of summaries and transcripts.
#some examples should be hallucination free, some should have hallucinations
#we need to check our function to see how good we can get it at remaining truthful to source
#then we need to apply the same methodology to the qa and action items sections
#qa may need some additional data validation to ensure proper json output
def test_filter_summary_sentences():
    # Given
    lemur_summary = "Wildfires in Canada are making the air dirty in many places in the US. Smoke from the fires is traveling through the sky and making it hard to breathe in places like New York and Baltimore. The smoke has tiny pieces in it that can get inside your lungs if you breathe them. This can make you sick, especially kids and older adults. The pieces in the smoke are much more than normal and that's why the air is unhealthy. More people could get sick until the weather changes and moves the smoke away. Fires might happen more often in the future because of climate change, so dirty air could affect more places."

    model_name = "infgrad/stella-base-en-v2"
    transcript_id = "6mvfr2epvp-65bc-46dc-8b4e-b87487b5da4b"  # The provided ID

    # When
    new_summary, filtered_sentences = filter_summary_sentences(lemur_summary, transcript_id, model_name)
    
    # Then
    expected_filtered_sentences = ['The smoke has tiny pieces in it that can get inside your lungs if you breathe them.']
    assert set(filtered_sentences) == set(expected_filtered_sentences), f"Expected {expected_filtered_sentences} but got {filtered_sentences}"