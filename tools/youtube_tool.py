from youtube_search import YoutubeSearch
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import xml.etree.ElementTree as ET
import colorama
from colorama import Fore, Style
import time
from collections import defaultdict

colorama.init(autoreset=True)

def slugify_filename(title):
    """Convert a title to a filename-friendly format."""
    title = re.sub(r'[^\w\s-]', '', title, flags=re.UNICODE)
    title = re.sub(r'[-\s]+', '_', title).strip('_')
    return title

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info_from_keyword(keyword, max_results=5):
    """Search YouTube for videos and extract their information."""
    try:
        results = YoutubeSearch(keyword, max_results=max_results).to_dict()
        videos = []
        
        for result in results:
            try:
                video_url = f"https://youtube.com{result['url_suffix']}"
                video_title = result['title']
                
                # Extract video ID using improved method
                video_id = extract_video_id(video_url)
                
                if not video_id:
                    # Try to get ID from result dict if available
                    if 'id' in result:
                        video_id = result['id']
                    else:
                        logging.warning(f"Could not extract video ID from: {video_url}")
                        continue
                
                # Validate video ID format (11 characters, alphanumeric + _ -)
                if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
                    logging.warning(f"Invalid video ID format: {video_id}")
                    continue
                
                videos.append({
                    'title': video_title,
                    'url': video_url,
                    'id': video_id
                })
                
            except Exception as e:
                logging.error(f"Error processing video result: {e}")
                continue
        
        if not videos:
            logging.warning(f"No valid videos found for keyword '{keyword}'")
        
        return videos
        
    except Exception as e:
        logging.error(f"Error searching for videos with keyword '{keyword}': {e}")
        return []

def get_youtube_transcript_aggressive(video_id, max_retries=5):
    """
    Aggressively retrieve transcript from YouTube video in ANY language and ANY condition.
    Will try every possible method to get transcript content.
    """
    if not video_id or not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
        logging.error(f"Invalid video ID: {video_id}")
        return None
    
    for attempt in range(max_retries):
        try:
            # Get ALL available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Strategy 1: Try to get ANY transcript (manual first, then auto-generated)
            transcripts_to_try = []
            
            # First, collect all manual transcripts
            for transcript in transcript_list:
                if not transcript.is_generated:
                    transcripts_to_try.append(transcript)
            
            # Then, collect all auto-generated transcripts
            for transcript in transcript_list:
                if transcript.is_generated:
                    transcripts_to_try.append(transcript)
            
            # Try each transcript
            for transcript in transcripts_to_try:
                try:
                    transcript_data = transcript.fetch()
                    if transcript_data and len(transcript_data) > 0:
                        text_segments = []
                        for entry in transcript_data:
                            if 'text' in entry and entry['text']:
                                text = entry['text'].strip()
                                if text:
                                    text_segments.append(text)
                        
                        if text_segments:
                            full_text = ' '.join(text_segments)
                            if len(full_text.strip()) > 10:  # Ensure meaningful content
                                logging.info(f"Got transcript for {video_id} in language: {transcript.language_code} (Generated: {transcript.is_generated})")
                                return full_text
                except Exception as e:
                    logging.debug(f"Failed to fetch transcript {transcript.language_code}: {e}")
                    continue
            
            # Strategy 2: Try to translate any available transcript to English
            try:
                for transcript in transcript_list:
                    try:
                        if transcript.is_translatable:
                            # Try to translate to English
                            translated = transcript.translate('en')
                            transcript_data = translated.fetch()
                            if transcript_data and len(transcript_data) > 0:
                                text_segments = []
                                for entry in transcript_data:
                                    if 'text' in entry and entry['text']:
                                        text = entry['text'].strip()
                                        if text:
                                            text_segments.append(text)
                                
                                if text_segments:
                                    full_text = ' '.join(text_segments)
                                    if len(full_text.strip()) > 10:
                                        logging.info(f"Got translated transcript for {video_id} from {transcript.language_code} to English")
                                        return full_text
                    except Exception as e:
                        logging.debug(f"Failed to translate transcript {transcript.language_code}: {e}")
                        continue
            except Exception as e:
                logging.debug(f"Translation strategy failed: {e}")
            
            # Strategy 3: Try direct API call with various language codes
            common_languages = ['en', 'en-US', 'en-GB', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi']
            
            for lang in common_languages:
                try:
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    if transcript_data and len(transcript_data) > 0:
                        text_segments = []
                        for entry in transcript_data:
                            if 'text' in entry and entry['text']:
                                text = entry['text'].strip()
                                if text:
                                    text_segments.append(text)
                        
                        if text_segments:
                            full_text = ' '.join(text_segments)
                            if len(full_text.strip()) > 10:
                                logging.info(f"Got transcript for {video_id} using direct API in language: {lang}")
                                return full_text
                except Exception as e:
                    logging.debug(f"Direct API failed for language {lang}: {e}")
                    continue
            
            # Strategy 4: Try with no language specification (let API decide)
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
                if transcript_data and len(transcript_data) > 0:
                    text_segments = []
                    for entry in transcript_data:
                        if 'text' in entry and entry['text']:
                            text = entry['text'].strip()
                            if text:
                                text_segments.append(text)
                    
                    if text_segments:
                        full_text = ' '.join(text_segments)
                        if len(full_text.strip()) > 10:
                            logging.info(f"Got transcript for {video_id} using default API call")
                            return full_text
            except Exception as e:
                logging.debug(f"Default API call failed: {e}")
            
        except TranscriptsDisabled:
            logging.warning(f"Transcripts are disabled for video {video_id}")
            return None
        except NoTranscriptFound:
            logging.warning(f"No transcript found for video {video_id}")
            return None
        except Exception as e:
            logging.debug(f"Attempt {attempt + 1} failed for video {video_id}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    logging.error(f"Failed to get transcript for video {video_id} after {max_retries} attempts with all strategies")
    return None

def get_all_available_transcripts_info(video_id):
    """Get detailed information about all available transcripts for debugging."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_transcripts = []
        
        for transcript in transcript_list:
            try:
                transcript_info = {
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                }
                available_transcripts.append(transcript_info)
            except Exception as e:
                logging.debug(f"Error getting transcript info: {e}")
                continue
        
        return available_transcripts
    except Exception as e:
        logging.debug(f"Error listing transcripts for {video_id}: {e}")
        return []

# Initialize the sentence transformer model for embedding generation
try:
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure you have an internet connection or the model is cached locally.")
    raise

# Initialize Qdrant client in local mode with persistent storage
qdrant_client = QdrantClient(path="./qdrant_db")

# Collection name for YouTube transcripts
COLLECTION_NAME = "youtube_transcripts"

def ensure_collection_exists():
    """Ensure the Qdrant collection exists, creating it if needed."""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            logging.info(f"Creating new Qdrant collection: {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            logging.info(f"Created new collection: {COLLECTION_NAME}")
        else:
            logging.info(f"Using existing Qdrant collection: {COLLECTION_NAME}")
    except Exception as e:
        logging.error(f"Error ensuring collection exists: {e}")
        raise

def add_videos_to_qdrant(videos_info, keyword=None, max_replacements=2, chunk_size=2000, chunk_overlap=200):
    """
    Get transcripts for provided videos, chunk them, and add to Qdrant.
    Uses aggressive transcript retrieval to get ANY available transcript.
    """
    if not videos_info:
        logging.warning("No video information provided to add_videos_to_qdrant.")
        return 0

    ensure_collection_exists()
    
    total_chunks_added = 0
    videos_with_transcripts = 0
    processed_video_ids = set()
    additional_videos_fetched = 0
    
    # Process all initial videos
    for video in videos_info:
        video_id = video['id']
        processed_video_ids.add(video_id)
        
        # Show available transcripts for debugging
        available_transcripts = get_all_available_transcripts_info(video_id)
        if available_transcripts:
            logging.info(f"Available transcripts for {video_id}: {available_transcripts}")
        
        # Use aggressive transcript retrieval
        transcript = get_youtube_transcript_aggressive(video_id)
        
        if transcript and len(transcript.strip()) > 50:
            print(f"{Fore.GREEN}✓ {Style.RESET_ALL}{video['title']} (Length: {len(transcript)} chars)")
            videos_with_transcripts += 1
            
            # Create overlapping chunks
            chunks_for_this_video = 0
            for i in range(0, len(transcript), chunk_size - chunk_overlap):
                chunk = transcript[i:i + chunk_size]
                if len(chunk.strip()) < 50:
                    continue
                
                # Create robust point ID
                point_id = abs(hash(f"{video_id}_{i}_{chunk[:100]}")) % (2**63 - 1)
                
                metadata = {
                    "title": video['title'],
                    "url": video['url'],
                    "video_id": video_id,
                    "chunk_index": i,
                    "original_position": i,
                    "content": chunk.strip()
                }
                
                try:
                    # Generate embedding
                    embedding = model.encode(chunk, show_progress_bar=False)
                    
                    # Add to Qdrant
                    qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.tolist(),
                                payload=metadata
                            )
                        ]
                    )
                    
                    total_chunks_added += 1
                    chunks_for_this_video += 1
                    
                except Exception as e:
                    logging.error(f"Error adding chunk to Qdrant: {e}")
                    continue
            
            logging.info(f"Added {chunks_for_this_video} chunks for '{video['title']}'")
            
        else:
            print(f"{Fore.RED}✗ {Style.RESET_ALL}{video['title']} - No transcript available")
            
            # Try to get replacement videos
            if keyword and additional_videos_fetched < max_replacements:
                logging.info(f"Searching for replacement videos for '{video['title']}'")
                
                replacement_videos = get_video_info_from_keyword(keyword, max_results=max_replacements * 3)
                replacement_videos = [v for v in replacement_videos if v['id'] not in processed_video_ids]
                
                for replacement in replacement_videos:
                    if additional_videos_fetched >= max_replacements:
                        break
                    
                    replacement_transcript = get_youtube_transcript_aggressive(replacement['id'])
                    
                    if replacement_transcript and len(replacement_transcript.strip()) > 50:
                        print(f"{Fore.GREEN}✓ {Style.RESET_ALL}[REPLACEMENT] {replacement['title']} (Length: {len(replacement_transcript)} chars)")
                        processed_video_ids.add(replacement['id'])
                        videos_with_transcripts += 1
                        additional_videos_fetched += 1
                        
                        # Process replacement transcript
                        chunks_for_replacement = 0
                        for i in range(0, len(replacement_transcript), chunk_size - chunk_overlap):
                            chunk = replacement_transcript[i:i + chunk_size]
                            if len(chunk.strip()) < 50:
                                continue
                            
                            point_id = abs(hash(f"{replacement['id']}_{i}_{chunk[:100]}")) % (2**63 - 1)
                            
                            metadata = {
                                "title": replacement['title'],
                                "url": replacement['url'],
                                "video_id": replacement['id'],
                                "chunk_index": i,
                                "original_position": i,
                                "content": chunk.strip()
                            }
                            
                            try:
                                embedding = model.encode(chunk, show_progress_bar=False)
                                
                                qdrant_client.upsert(
                                    collection_name=COLLECTION_NAME,
                                    points=[
                                        models.PointStruct(
                                            id=point_id,
                                            vector=embedding.tolist(),
                                            payload=metadata
                                        )
                                    ]
                                )
                                
                                total_chunks_added += 1
                                chunks_for_replacement += 1
                                
                            except Exception as e:
                                logging.error(f"Error adding replacement chunk to Qdrant: {e}")
                                continue
                        
                        logging.info(f"Added {chunks_for_replacement} chunks for replacement '{replacement['title']}'")
                        break
                    else:
                        print(f"{Fore.RED}✗ {Style.RESET_ALL}[REPLACEMENT FAILED] {replacement['title']} - No transcript")
    
    logging.info(f"Summary: {videos_with_transcripts} videos with transcripts, {total_chunks_added} total chunks added")
    if additional_videos_fetched > 0:
        logging.info(f"Added {additional_videos_fetched} replacement videos")
    
    return total_chunks_added

def search_youtube_transcripts(query, n_results_per_video=3, fetch_limit=100):
    """Perform semantic search on stored transcripts."""
    try:
        # Generate query embedding
        query_embedding = model.encode(query, show_progress_bar=False)
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=fetch_limit
        )
        
        if not search_results:
            logging.info(f"No search results found for query: '{query}'")
            return {}
        
        # Group by video_id
        video_results = defaultdict(list)
        for result in search_results:
            vid = result.payload.get('video_id')
            if vid:
                video_results[vid].append(result)
        
        # Format results
        formatted_results = {}
        for vid, results_list in video_results.items():
            sorted_results = sorted(results_list, key=lambda r: r.score, reverse=True)[:n_results_per_video]
            
            if sorted_results:
                video_title = sorted_results[0].payload.get('title', 'Unnamed Video')
                formatted_results[vid] = {
                    "title": video_title,
                    "chunks": []
                }
                
                for result in sorted_results:
                    formatted_results[vid]["chunks"].append({
                        "score": result.score,
                        "content": result.payload.get('content', ''),
                        "url": result.payload.get('url', ''),
                        "original_position": result.payload.get('original_position', 0)
                    })
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return {}

def process_and_store_youtube_videos(keyword, max_results=5, chunk_size=1000, chunk_overlap=100, max_replacements=2):
    """Main function to process and store YouTube videos."""
    try:
        logging.info(f"Processing YouTube videos for keyword: '{keyword}'")
        
        # Get videos
        videos = get_video_info_from_keyword(keyword, max_results=max_results)
        if not videos:
            msg = f"No videos found for keyword '{keyword}'"
            logging.warning(msg)
            return False, msg, []
        
        # Process transcripts
        chunks_added = add_videos_to_qdrant(
            videos,
            keyword=keyword,
            max_replacements=max_replacements,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if chunks_added == 0:
            msg = f"No transcripts retrieved for keyword '{keyword}'"
            logging.warning(msg)
            return False, msg, videos
        
        msg = f"Successfully processed {len(videos)} videos, added {chunks_added} chunks to database"
        logging.info(msg)
        return True, msg, videos
        
    except Exception as e:
        msg = f"Error processing videos: {str(e)}"
        logging.error(msg)
        return False, msg, []

def count_documents():
    """Count documents in the collection."""
    try:
        count_result = qdrant_client.count(collection_name=COLLECTION_NAME, exact=True)
        return count_result.count
    except Exception as e:
        logging.error(f"Error counting documents: {e}")
        return 0

def main():
    """Main demonstration function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test with a topic
    search_topic = "machine learning basics"
    print(f"\n=== Processing videos about '{search_topic}' ===")
    
    success, message, videos = process_and_store_youtube_videos(
        search_topic,
        max_results=7,
        max_replacements=5
    )
    
    print(f"\nResult: {message}")
    
    if success and videos:
        print(f"\nProcessed {len(videos)} videos:")
        for i, video in enumerate(videos, 1):
            print(f"  {i}. {video['title']}")
            print(f"     ID: {video['id']}")
            print(f"     URL: {video['url']}")
            
            # Show transcript info
            transcripts = get_all_available_transcripts_info(video['id'])
            if transcripts:
                print(f"     Transcripts: {len(transcripts)} available")
                for t in transcripts[:3]:  # Show first 3
                    print(f"       - {t['language']} ({'auto' if t['is_generated'] else 'manual'})")
            else:
                print(f"     Transcripts: None available")
            print()
    
    # Check database
    doc_count = count_documents()
    print(f"\nTotal documents in database: {doc_count}")
    
    # Test search
    if doc_count > 0:
        query = "neural networks"
        print(f"\n=== Searching for '{query}' ===")
        results = search_youtube_transcripts(query, n_results_per_video=2)
        
        if results:
            print(f"Found results in {len(results)} videos:")
            for video_id, data in results.items():
                print(f"\nVideo: {data['title']}")
                for i, chunk in enumerate(data['chunks'], 1):
                    print(f"  Result {i} (Score: {chunk['score']:.4f}):")
                    print(f"  {chunk['content'][:300]}...")
                    print(f"  URL: {chunk['url']}")
                    print()
        else:
            print("No search results found")

if __name__ == "__main__":
    main()