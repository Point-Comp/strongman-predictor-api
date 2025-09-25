# 1_data_preparation.py (Complete & Final Version)
import pandas as pd
import numpy as np

def parse_performance(perf_str):

    perf_str = str(perf_str).strip().lower()
    reps, time_s, distance_m = 0.0, 0.0, 0.0
    try:
        if ' in ' in perf_str:
            parts = perf_str.replace(' s', '').split(' in ')
            reps = float(parts[0])
            time_s = float(parts[1])
        elif 'reps' in perf_str:
            reps = float(perf_str.replace(' reps', ''))
        elif ' s' in perf_str:
            time_s = float(perf_str.replace(' s', ''))
        elif ' m' in perf_str:
            distance_m = float(perf_str.replace(' m', ''))
    except (ValueError, IndexError):
        pass
    return {'Reps': reps, 'Time_s': time_s, 'Distance_m': distance_m}

def categorize_event(name):

    name = str(name).lower()
    if 'press' in name or 'log lift' in name: return 'Pressing'
    if 'deadlift' in name: return 'Deadlift'
    if 'walk' in name or 'yoke' in name or 'carry' in name: return 'Carrying'
    if 'stones' in name: return 'Stones'
    if 'toss' in name or 'flip' in name: return 'Explosive'
    return 'Other'

def run_preparation(events_path, athletes_path, contests_path, output_cleaned_path, output_profile_path):
    
    print("Step 1: Loading and Merging Data...")
    events_df = pd.read_csv(events_path)
    athletes_df = pd.read_csv(athletes_path)
    contests_df = pd.read_csv(contests_path)

    try:
        merged_df = pd.merge(events_df, contests_df, on=['athlete_name', 'date', 'contest'], suffixes=('_event', '_contest'))
        df = pd.merge(merged_df, athletes_df, left_on='athlete_name', right_on='full_name')
        df = df.rename(columns={'placing_event': 'event_placing', 'placing_contest': 'final_placing'})
        df = df.drop(columns=['full_name', 'url', 'athlete_url_event', 'athlete_url_contest'])
    except Exception as e:
        print(f"Merge Error: {e}. Please check your CSV headers and key columns.")
        return

    print("Data successfully merged. Starting cleaning process...")
    
    print("Step 2: Parsing and Cleaning Data...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'event_placing', 'final_placing'], inplace=True)
    
    df[['event_place', 'competitor_count']] = df['event_placing'].str.split(' of ', expand=True).astype(int)
    
    perf_data = df['result'].apply(parse_performance)
    df_perf = pd.json_normalize(perf_data)
    df = pd.concat([df.reset_index(drop=True), df_perf.reset_index(drop=True)], axis=1)

    df.columns = [col.lower().strip() for col in df.columns]

    print("Step 3: Engineering Base Features...")
    df['perf_score'] = 1 - ((df['event_place'] - 1) / (df['competitor_count'] - 1))
    df['event_category'] = df['event'].apply(categorize_event)
    df['perf_score'] = df['perf_score'].fillna(1.0) 
    df['final_placing'] = df['final_placing'].str.split(' of ', expand=True)[0].astype(int)

    df.to_csv(output_cleaned_path, index=False)
    print(f"Cleaned event data saved to '{output_cleaned_path}'")

    print("Step 4: Creating Athlete Profiles...")
    athlete_event_features = df.groupby(['athlete_name', 'event_category']).agg(
        avg_reps=('reps', 'mean'),
        avg_time=('time_s', 'mean'),
        avg_perf_score=('perf_score', 'mean')
    ).reset_index()

    athlete_profile = athlete_event_features.pivot_table(
        index='athlete_name',
        columns='event_category',
        values=['avg_reps', 'avg_time', 'avg_perf_score']
    )
    athlete_profile.columns = ['_'.join(col).strip().lower() for col in athlete_profile.columns.values]
    
    print("Step 5: Creating speed scores...")
    time_cols = [col for col in athlete_profile.columns if 'avg_time' in col]
    for col in time_cols:
        athlete_profile[col] = np.where(athlete_profile[col] > 0, 1 / athlete_profile[col], 0)
        new_name = col.replace('avg_time', 'speed_score')
        athlete_profile.rename(columns={col: new_name}, inplace=True)

    print("Step 6: Calculating 1-Rep Max features...")
    df['weight_kg'] = pd.to_numeric(df['event'].str.extract(r'\((\d+\.?\d*)\s*kg\)')[0], errors='coerce')
    one_rep_df = df[df['result'].str.strip() == '1 rep'].copy()
    max_lift_features = one_rep_df.groupby(['athlete_name', 'event_category'])['weight_kg'].max().unstack()
    
    if 'Deadlift' in max_lift_features.columns:
        max_lift_features.rename(columns={'Deadlift': 'one_rep_max_deadlift'}, inplace=True)
    if 'Pressing' in max_lift_features.columns:
        max_lift_features.rename(columns={'Pressing': 'one_rep_max_pressing'}, inplace=True)
        
    max_lift_features = max_lift_features[[col for col in ['one_rep_max_deadlift', 'one_rep_max_pressing'] if col in max_lift_features.columns]]

    athlete_profile = pd.merge(athlete_profile, max_lift_features, on='athlete_name', how='left')
    
    athlete_profile = athlete_profile.fillna(0)

    athlete_profile.to_csv(output_profile_path)
    print(f"Athlete profiles with all features saved to '{output_profile_path}'")

if __name__ == '__main__':
    EVENTS_CSV = 'events.csv'
    ATHLETES_CSV = 'athletes.csv'
    CONTESTS_CSV = 'contests.csv'
    
    CLEANED_DATA_CSV = 'cleaned_strongman_data.csv'
    ATHLETE_PROFILES_CSV = 'athlete_profiles.csv'
    
    run_preparation(EVENTS_CSV, ATHLETES_CSV, CONTESTS_CSV, CLEANED_DATA_CSV, ATHLETE_PROFILES_CSV)