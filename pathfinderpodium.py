import pandas as pd
from collections import deque

def build_podium_graph(filepath='contests.csv'):
    """
    Loads the contest data and builds a graph of podium connections.
    Returns BOTH the graph and a sorted list of all unique athlete names.
    """
    print("Loading data and building podium graph...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure it's in the same folder.")
        # Return two values in the error case to prevent crashing
        return None, None

    # --- Data Cleaning and Preparation ---
    df['PlacementNumber'] = df['placing'].str.extract('(\d+)').astype(float)
    podium_df = df[df['PlacementNumber'].isin([1, 2, 3])].copy()
    
    # Get a sorted list of all unique athletes who have been on a podium
    all_athletes = sorted(podium_df['athlete_name'].unique())

    # --- Graph Construction ---
    graph = {}
    podiums_by_contest = podium_df.groupby(['contest', 'date'])['athlete_name'].apply(list)

    for contest_podium_list in podiums_by_contest:
        for i in range(len(contest_podium_list)):
            for j in range(i + 1, len(contest_podium_list)):
                athlete1, athlete2 = contest_podium_list[i], contest_podium_list[j]
                
                # Find the specific contest name and date for this connection
                # This is a bit complex, but finds the index (contest, date) for the current list of athletes
                contest_group = podiums_by_contest[podiums_by_contest.apply(lambda x: athlete1 in x and athlete2 in x)].index[0]
                contest_name, contest_date = contest_group[0], contest_group[1]
                connection_details = {'contest': contest_name, 'date': contest_date}

                graph.setdefault(athlete1, {})[athlete2] = connection_details
                graph.setdefault(athlete2, {})[athlete1] = connection_details
    
    print("Graph build complete.")
    # This now correctly returns two values
    return graph, all_athletes


def find_shortest_path(graph, start_athlete, end_athlete):
    """
    Finds the shortest path between two athletes using Breadth-First Search (BFS).
    """
    if start_athlete not in graph or end_athlete not in graph:
        return None

    queue = deque([[{'name': start_athlete, 'connections': {}}]]) 
    visited = {start_athlete}

    while queue:
        path = queue.popleft()
        current_node_name = path[-1]['name']

        if current_node_name == end_athlete:
            return path

        if current_node_name in graph:
            for neighbor_name, connection_details in graph[current_node_name].items():
                if neighbor_name not in visited:
                    visited.add(neighbor_name)
                    new_path = list(path)
                    
                    path_step = {
                        'name': neighbor_name,
                        'connections': {
                            current_node_name: connection_details
                        }
                    }
                    new_path.append(path_step)
                    queue.append(new_path)
    return None
