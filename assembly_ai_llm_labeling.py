import assemblyai as aai 
from openai import OpenAI
import os
import pandas as pd 
from sqlalchemy import create_engine
import json 
from tqdm import tqdm
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import traceback
from datetime import datetime 

load_dotenv()

class TranscribeLabelPipeline: 

    def __init__(self, audio_file=None, transcript_id=None):

        self.audio_file = audio_file
        self.transcript_id = transcript_id
        self.transcript = None 
        self.transcriber = None 
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
        self.aai_key = os.getenv("AAI_API_KEY")
        self.aai_paragraph_endpoint = f"https://api.assemblyai.com/v2/transcript/{self.transcript_id}/paragraphs"
        self.aai_transcript_endpoint = f"https://api.assemblyai.com/v2/transcript/{self.transcript_id}"

    def get_transcript(self):
        if not self.transcript_id:
            try: 
                aai.settings.api_key = self.aai_key
                config = aai.TranscriptionConfig(iab_categories=False)
                
                self.transcriber = aai.Transcriber().transcribe(self.audio_file, config)
                print("Transcriber object defined")

                self.transcript_id = self.transcriber.id 
                print("Obtained Transcript ID")
                self.transcript = self.transcriber.text 
                print("Transcript obtained!")

                return self 
            
            except Exception as e: 
                print("No transcript ID - please attach an audio file for transcription")
                return self 

        else: 
            print("Transcript ID already present")

            print("Using AAI transcript endpoint to fetch transcript text")

            transcript_response = requests.get(self.aai_transcript_endpoint, headers={'authorization': self.aai_key})
            transcript_data = transcript_response.json()

            if transcript_data.get('status') == 'completed':
                self.transcript = transcript_data['text']
                print("Transcript obtained!")

            else: 
                print(f"Transcript not ready, status: {transcript_data.get('status')}")

            return self 

    def get_paragraphs(self):
        if self.transcriber: 
            paragraphs_response = self.transcriber.get_paragraphs()

            return {'paragraphs': [{'text': p.text, 'start': p.start, 'end': p.end} for p in paragraphs_response]}
        
        elif self.transcript_id: 
            headers = {'authorization': self.aai_key}
            paragraphs_response = requests.get(self.aai_paragraph_endpoint, headers=headers)

            return paragraphs_response.json()

        else: 
            print("No Transcript ID for paragraphs endpoint")
            return None 

    def get_transcript_summary(self): 
        if self.transcript_id: 

            prompt = f"""
            Please summarize this podcast transcript in fewer than 5 paragraphs.

            {self.transcript}
            """

            try:
                summary_response = self.client.responses.parse(
                    model="o4-mini-2025-04-16",
                    input=[
                        {
                            "role": "system",
                            "content": "You are an expert at summarizing podcast transcripts. This summary will be used by AI to assess if excerpted transcript paragraphs are advertisements or podcast content."
                            },
                        {"role": "user", "content": prompt},
                    ],
                )

                return summary_response.output_text

            except Exception as e: 
                print("Failed to summarize podcast transcript")
                print(traceback.format_exc())
                return None 

    def analyze_segment_for_ads(self, transcript_summary, paragraph, paragraph_start, paragraph_end):

        class PodcastDataLabels(BaseModel):
            start_time: float
            end_time: float
            confidence: int
            label: str 
            reasoning: str

        prompt = f"""
    This prompt includes the summary of the transcript of a podcast, in addition to a singular paragraph of the transcript with assosciated timestamps (start time and end time of the paragraph). 

    Use the context of the transcript summary in addition to your background knowledge about advertisements (ads) to determine if the paragraph is podcast content or an advertisement (ad).

    Use the confidence scale described below in addition to the other desired output responses. 

    Confidence: 1-10 (10 = definitely an ad, 1 = definitely content and definitely not an ad).
    Label: "content", "ad"
    start_time: the beginning timestamp of the segment. 
    end_time: the ending timestamp of the segment.
    reasoning: Reason the segment is content or ad. Please keep the reasoning succint (less than 10 words). 

    Transcript: {transcript_summary}

    Transcript paragraph (start time: {paragraph_start} -- end time: {paragraph_end}):
    {paragraph}

    """

        try:
            response = self.client.responses.parse(
                model="o4-mini-2025-04-16",
                input=[
                    {
                        "role": "system",
                        "content": "You are an expert at identifying advertisements in podcast transcripts. Please convert your resonse to the given structure. Please remember to use the 1-10 confidence ratings as described in the user prompt."
                        },
                    {"role": "user", "content": prompt},
                ],
                text_format=PodcastDataLabels,
            )
            
            # Parse the JSON response
            parsed_result = response.output_parsed 

            result_dict = {
                'transcript_id': self.transcript_id,
                'start_time': parsed_result.start_time,
                'end_time': parsed_result.end_time,
                'confidence': parsed_result.confidence,
                'label': parsed_result.label,
                'label_dt': datetime.now(),
                'reasoning': parsed_result.reasoning
            }
            
            return result_dict
            
        except Exception as e:
            print(f"Error analyzing chunk: {e}")
            return None

    def analyze_episode_for_ads(self):
        
        print("Getting transcript...")
        self.get_transcript()
        print("Getting paragraphs...")
        paragraphs_data = self.get_paragraphs()

        num_paragrahs = len(paragraphs_data['paragraphs'])
        print(f"{num_paragrahs} paragraphs retrieved")

        print("Getting episode summary...")
        summary = self.get_transcript_summary()

        episode_results = []

        print("Labeling data...Beep Boop")
        for i in tqdm(range(num_paragrahs)): 

            segment = paragraphs_data['paragraphs'][i]

            segment_results = self.analyze_segment_for_ads(transcript_summary=summary, 
                                                           paragraph=segment['text'],
                                                           paragraph_start=segment['start'], 
                                                           paragraph_end=segment['end'])
            if segment_results:
                episode_results.append(segment_results)

        return episode_results # returns a list of dictionaries
    

class QueryDatabase: 

    def __init__(self):
        self.db_engine = create_engine(os.getenv("DB_CONN_STRING"))
        print("Built DB engine")

    def write_data(self, df, table_name): 

        try: 
            print(f"Attempting to write {len(df)} rows to {table_name}")

            with self.db_engine.connect() as conn: 
                df.to_sql(name=table_name, con=conn, if_exists='append', index=False, chunksize=1000, method='multi')

            print(f"Sucessfully wrote data to {table_name}")
            print("Disposing of Engine")

            self.db_engine.dispose()

            return self

        except Exception: 
            self.db_engine.dispose()
            print(f"Failed to write data to {table_name}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None
        
    def read_data(self, query):

        try: 
            print("Attempting to read data")
            with self.db_engine.connect() as conn: 
                df = pd.read_sql_query(sql=query, con=conn)

            print("Sucessfully read data")
            print("Disposing of Engine")

            return df
        
        except Exception as e:
            self.db_engine.dispose()
            print(f"Failed to read data: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            return None 


        
if __name__ == "__main__": 

    query_transcripts = "SELECT audio_file FROM transcript_log"

    q = QueryDatabase().read_data(query=query_transcripts)

    labeled_files = q.audio_file.to_list()

    training_data_path = "training_data/audio/"

    previously_transcribed_paths = set()

    for f in labeled_files:
        previously_transcribed_paths.add(os.path.join(training_data_path, f))



    previously_transcribed = ["training_data/audio/Dead-Awake_1.mp3",
                               "training_data/audio/How-economists-and-TikTok-know-if-a-recession-is-coming_1.mp3",
                               "training_data/audio/317-Nicole-Rust-on-Why-Neuroscience-Hasnt-Solved-Brain-Disorders_0.mp3", 
                               "training_data/audio/Advice-Line-with-Steve-Holmes-of-goba-Sports-Group_0.mp3", 
                               "training_data/audio/202-Steve-Kwast-How-China-is-Mining-the-Moon-and-Weaponizing-Space_1.mp3", 
                               "training_data/audio/The-secret-world-behind-those-scammy-text-messages_0.mp3", 
                               "training_data/audio/203-Dave-Mustaine-Megadeth-Co-Founder-Frontman_0.mp3",
                               "training_data/audio/2328-Luke-Caverns_1.mp3",
                               "training_data/audio/Ciara-Dont-Let-Him-Waste-Your-Time-FBF_0.mp3"]

    for audio_file in os.listdir(training_data_path):
        full_path = os.path.join(training_data_path, audio_file)

        if full_path in previously_transcribed_paths or full_path in previously_transcribed:
            print(f"{audio_file} already transcribed")
            continue

        else: 
            print(f"Transcribing and labeling: {audio_file}")

            transcript_log = [{'transcript_id': None,
                            'transcription_date': datetime.now(), 
                            'audio_file': audio_file
                            }]

            pipe = TranscribeLabelPipeline(transcript_id=transcript_log[0]['transcript_id'],
                                            audio_file=full_path) # need to read the full path for transcription
            
            episode_results = pipe.analyze_episode_for_ads()

            # extract the transcript id from the episode results dict 
            # this loop operates on 1 audio file so the transcript id is the same 
            retrieved_transcript_id = episode_results[0]['transcript_id']
            transcript_log[0]['transcript_id'] = retrieved_transcript_id

            log = pd.DataFrame.from_dict(data=transcript_log, orient='columns')

            QueryDatabase().write_data(df=log, table_name='transcript_log')

            df_results = pd.DataFrame.from_dict(data=episode_results, orient='columns')

            QueryDatabase().write_data(df=df_results, table_name='podcast_segment_labels')
