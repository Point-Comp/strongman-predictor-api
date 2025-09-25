import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- Configuration ---
BASE_URL = "https://www.strongmanarchives.com/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# Using 20 concurrent threads for scraping. You can increase this if you have a very fast connection.
MAX_WORKERS = 20

def scrape_athlete_details(athlete_url, athlete_name):
    """
    Scrapes BOTH contest history and event lifts for a single athlete.
    This function is designed to be run in a separate thread.
    """
    contests = []
    events = []
    
    try:
        response = requests.get(athlete_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        # --- Scrape the Contests Table (id="CompTable") ---
        contest_table = soup.find('table', id='CompTable')
        if contest_table:
            tbody = contest_table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) == 6:
                        contests.append({
                            'athlete_name': athlete_name, 'date': cells[0].text.strip(),
                            'contest': cells[1].text.strip(), 'contest_type': cells[2].text.strip(),
                            'division': cells[3].text.strip(), 'location': cells[4].text.strip(),
                            'placing': cells[5].text.strip(), 'athlete_url': athlete_url
                        })

        # --- Scrape the Events/Lifts Table (id="EventTable") ---
        event_table = soup.find('table', id='EventTable')
        if event_table:
            tbody = event_table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) == 5:
                        events.append({
                            'athlete_name': athlete_name, 'date': cells[0].text.strip(),
                            'contest': cells[1].text.strip(), 'event': cells[2].text.strip(),
                            'result': cells[3].text.strip(), 'placing': cells[4].text.strip(),
                            'athlete_url': athlete_url
                        })
    except requests.exceptions.RequestException:
        # Errors will be handled by not returning data for this athlete.
        # We avoid printing here to keep the progress bar clean.
        pass
    
    # No time.sleep() for maximum speed
    return contests, events

def get_all_athlete_links():
    """
    Phase 1: Sequentially scrape all list pages to gather athlete URLs.
    This is fast and avoids complex state in parallel execution.
    """
    page_number = 1
    athletes_to_scrape = []
    athlete_metadata = []
    seen_athlete_urls = set()

    while True:
        list_url = urljoin(BASE_URL, f"athletes.php?page={page_number}")
        print(f"Discovering athletes on page: {page_number}", end='\r')
        
        try:
            response = requests.get(list_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            athlete_rows = soup.find_all('tr', style="font-size:11px")

            if not athlete_rows:
                break # Finished all pages

            for row in athlete_rows:
                cells = row.find_all('td')
                link_tag = cells[1].find('a')
                if link_tag and link_tag.has_attr('href'):
                    athlete_url = urljoin(BASE_URL, link_tag['href'])

                    if athlete_url in seen_athlete_urls:
                        print("\nDetected repeat athlete, halting discovery phase.")
                        return athlete_metadata, athletes_to_scrape

                    seen_athlete_urls.add(athlete_url)
                    athlete_name = cells[0].text.strip()
                    
                    athlete_metadata.append({
                        'full_name': athlete_name, 'country': cells[3].text.strip(),
                        'active_years': cells[4].text.strip(), 'competitions': cells[5].text.strip(),
                        'url': athlete_url
                    })
                    athletes_to_scrape.append((athlete_url, athlete_name))
            
            page_number += 1
        except requests.exceptions.RequestException as e:
            print(f"\nError during discovery on page {page_number}: {e}")
            break

    print("\nDiscovery phase complete.")
    return athlete_metadata, athletes_to_scrape

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting Strongman Archives Scraper 🏋️ (Press Ctrl+C to stop and save)")
    start_time = time.time()
    
    all_athletes_metadata = []
    all_contests = []
    all_events = []

    try:
        # Phase 1: Discover all athlete URLs first.
        all_athletes_metadata, tasks = get_all_athlete_links()
        
        # Phase 2: Scrape details concurrently with a progress bar.
        print(f"Scraping details for {len(tasks)} athletes using {MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # The lambda function unpacks the tuple for the function arguments
            results = list(tqdm(executor.map(lambda p: scrape_athlete_details(*p), tasks), total=len(tasks)))

        # Process the results from all threads
        for contests_data, events_data in results:
            all_contests.extend(contests_data)
            all_events.extend(events_data)

    except (KeyboardInterrupt, SystemExit):
        print("\n\nScrape interrupted by user. Saving progress...")
    finally:
        if all_athletes_metadata:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n\nSaving data for {len(all_athletes_metadata)} athletes found in {duration:.2f} seconds.")
            print(f"  - Total contest entries: {len(all_contests)}")
            print(f"  - Total event/lift entries: {len(all_events)}")
            
            athletes_df = pd.DataFrame(all_athletes_metadata)
            contests_df = pd.DataFrame(all_contests)
            events_df = pd.DataFrame(all_events)
            
            athletes_df.to_csv('athletes.csv', index=False)
            contests_df.to_csv('contests.csv', index=False)
            events_df.to_csv('events.csv', index=False)
            
            print("\nData saved successfully to 'athletes.csv', 'contests.csv', and 'events.csv' 📈")
        else:
            print("\nNo data was collected before stopping.")

