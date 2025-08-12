import os
import re
import requests
import feedparser
import numpy as np
from datetime import datetime

class PodcastDataCollector:
    """
    class to collect and process podcast data for training
    """
    
    def __init__(self, output_dir="training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/audio", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
        
    def download_podcast_episodes(self, rss_urls, max_episodes_per_feed=5):
        """
        Download podcast episodes from RSS feeds
        
        Args:
            rss_urls: List of podcast RSS feed URLs
            max_episodes_per_feed: How many recent episodes to download per podcast
            
        Returns:
            List of downloaded episode info
        """
        episodes = []
        
        for rss_url in rss_urls:
            print(f"Processing feed: {rss_url}")
            
            # Parse the RSS feed to get episode information
            feed = feedparser.parse(rss_url)
            
            # Get the most recent episodes (they're usually in chronological order)
            for i, entry in enumerate(feed.entries[:max_episodes_per_feed]):
                try:
                    # Find the audio URL - usually in enclosures
                    audio_url = None
                    if hasattr(entry, 'enclosures') and entry.enclosures:
                        # Look for audio files (mp3, m4a, etc.)
                        for enclosure in entry.enclosures:
                            if 'audio' in enclosure.type:
                                audio_url = enclosure.href
                                break
                    
                    if not audio_url:
                        print(f"No audio URL found for episode: {entry.title}")
                        continue
                    
                    # Create a safe filename
                    safe_title = re.sub(r'[^\w\s-]', '', entry.title).strip()
                    safe_title = re.sub(r'[-\s]+', '-', safe_title)
                    filename = f"{safe_title}_{i}.mp3"
                    filepath = os.path.join(self.output_dir, "audio", filename)
                    
                    # Download the audio file
                    print(f"Downloading: {entry.title}")
                    response = requests.get(audio_url, stream=True)
                    response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Store episode metadata
                    episode_info = {
                        'title': entry.title,
                        'filename': filename,
                        'filepath': filepath,
                        'description': getattr(entry, 'description', ''),
                        'published': getattr(entry, 'published', ''),
                        'podcast_title': feed.feed.title
                    }
                    episodes.append(episode_info)
                    
                except Exception as e:
                    print(f"Error downloading episode {entry.title}: {e}")
                    continue
        
        return episodes


# Example usage and high-value podcast feeds for training data
def get_starter_podcast_feeds():
    """
    Returns a list of podcast RSS feeds that are good for training data
    These podcasts typically have clear ad patterns and good audio quality
    """
    
    return [
        # These are example feeds - you'll want to find current, working RSS URLs
        # "https://feeds.simplecast.com/qm_9xx0g",  # Crime Junkie
        # "https://feeds.megaphone.fm/GLT1412515089",     # Joe Rogan Show
        # "https://rss.art19.com/how-i-built-this",        # How I built this
        # "https://lexfridman.com/feed/podcast/", # lex fridman
        # "https://feeds.megaphone.fm/hubermanlab", # huberman lab 
        # "https://feeds.megaphone.fm/thispastweekend", # this past weekend w theo von
        # "https://feeds.megaphone.fm/good-hang-with-amy-poehler", # good hang 
        # "https://feeds.simplecast.com/54nAGcIl", # the daily NYT
        # "https://feeds.megaphone.fm/STU8858464340", # something scary
        # "https://feeds.simplecast.com/mKn_QmLS", # call her daddy
        # "https://feeds.megaphone.fm/unplannedpodcast" # the unplanned podcast
        # "https://rss.acast.com/the-gargle", 
        # "https://rss.art19.com/sean-carrolls-mindscape", 
        # "https://feeds.simplecast.com/82FI35Px", 
        # "https://feeds.simplecast.com/54nAGcIl", 
        "https://feeds.npr.org/510333/podcast.xml", 
        "https://feeds.npr.org/510289/podcast.xml", 
        "https://feeds.npr.org/381444908/podcast.xml", # fresh air 
        "https://feeds.simplecast.com/kwWc0lhf" # hidden brain
    ]


# Main execution example
if __name__ == "__main__":
    # Initialize the data collector
    collector = PodcastDataCollector()
    
    # Get some podcast feeds to work with
    podcast_feeds = get_starter_podcast_feeds()
    
    print("Step 1: Downloading podcast episodes...")
    episodes = collector.download_podcast_episodes(podcast_feeds, max_episodes_per_feed=3)
    
    print(f"Downloaded {len(episodes)} episodes")
