import praw
import json
from datetime import datetime, timezone
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Reddit API Credentials
CLIENT_ID = "nx8eatNC3ukmn5jK_z-RYg"
CLIENT_SECRET = "WniZSFznj5V8BQ8yYxLamqbHmelZTQ"
USERNAME = "Gaurav_Pawar_1349"
PASSWORD = "@Dahyane1349"
USER_AGENT = "Extraction"

# Authenticate with Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    username=USERNAME,
    password=PASSWORD,
    user_agent=USER_AGENT
)

def extract_reddit_posts(subreddit_name, start_date, end_date, output_file, languages):
    """
    Extracts Reddit posts related to complaints in specified languages and saves to JSON.

    Args:
        subreddit_name (str): Subreddit name (e.g., "india").
        start_date (str): Start date (e.g., "2025-01-01").
        end_date (str): End date (e.g., "2025-02-28").
        output_file (str): Output JSON file path.
        languages (list): List of language codes to filter (e.g., ["mr", "hi", "en"]).
    """
    # Convert dates to timestamps
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Fetch posts from subreddit
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    for post in subreddit.search("complaint OR grievance OR issue", time_filter="year", limit=None):
        # Filter by date range
        if start_timestamp <= post.created_utc <= end_timestamp:
            # Combine title and selftext for language detection
            text_to_detect = f"{post.title} {post.selftext}"
            try:
                detected_language = detect(text_to_detect)
            except LangDetectException:
                detected_language = "unknown"
            
            if detected_language in languages:
                post_info = {
                    "complaintid": post.id,
                    "title": post.title,
                    "description": post.selftext,
                    "date": datetime.fromtimestamp(post.created_utc, timezone.utc).strftime('%Y-%m-%d'),
                    "author": post.author.name if post.author else "Deleted",
                    "url": post.url,
                    "location": "India"  # Assuming location is India for all posts
                }
                posts_data.append(post_info)

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(posts_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(posts_data)} posts to {output_file}")

# Example usage
extract_reddit_posts(
    subreddit_name="india",
    start_date="2025-01-01",
    end_date="2025-03-28",
    output_file="india_complaints_reddit2.json",
    languages=["mr", "hi", "en"]  # Marathi, Hindi, and English language codes
)