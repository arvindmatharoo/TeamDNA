import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

st.set_page_config(layout="wide", page_title="⚽ StatsBomb Open Data Explorer")

# ==========================
# ---- LOAD COMPETITIONS ----
# ==========================
st.title("⚽ StatsBomb Open Data Explorer")

@st.cache_data
def load_competitions():
    url = "https://raw.githubusercontent.com/statsbomb/open-data/refs/heads/master/data/competitions.json"
    response = requests.get(url)
    competitions = response.json()
    return pd.DataFrame(competitions)

df_competitions = load_competitions()
st.subheader("Available Competitions")
st.dataframe(df_competitions[['competition_id','season_id','competition_name','season_name']])

# ---- SELECT COMPETITION & SEASON ----
comp_name = st.text_input("Enter part of competition name (e.g. 'World Cup')", "")
season_name = st.text_input("Enter part of season name (e.g. '2018')", "")

if comp_name and season_name:
    filtered = df_competitions[
        df_competitions['competition_name'].str.contains(comp_name, case=False) &
        df_competitions['season_name'].str.contains(season_name, case=False)
    ]
    if filtered.empty:
        st.warning("No matching competitions found. Check your spelling.")
    else:
        st.success("Matching competitions found ✅")
        st.dataframe(filtered[['competition_id','season_id','competition_name','season_name']])
        
        comp_id = st.selectbox("Select Competition ID", filtered['competition_id'].unique())
        season_id = st.selectbox("Select Season ID", filtered['season_id'].unique())
        
        st.session_state["comp_id"] = comp_id
        st.session_state["season_id"] = season_id

# ==========================
# ---- LOAD MATCHES --------
# ==========================
def load_matches(comp_id, season_id):
    url_matches = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{comp_id}/{season_id}.json"
    matches = requests.get(url_matches).json()
    df_matches = pd.DataFrame(matches)
    # Extract nested fields
    df_matches['home_team_name'] = df_matches['home_team'].apply(lambda x: x['home_team_name'])
    df_matches['away_team_name'] = df_matches['away_team'].apply(lambda x: x['away_team_name'])
    df_matches['stage'] = df_matches['competition_stage'].apply(lambda x: x['name'])
    df_matches['match_label'] = df_matches['home_team_name'] + " vs " + df_matches['away_team_name']
    return df_matches

if "comp_id" in st.session_state and "season_id" in st.session_state:
    comp_id = st.session_state["comp_id"]
    season_id = st.session_state["season_id"]
    df_matches = load_matches(comp_id, season_id)
    
    # ---- CALCULATE WINNERS ----
    def get_knockout_winner(row):
        knockout_stages = ['Round of 16','Quarter-finals','Semi-finals','Final']
        if row['stage'] in knockout_stages:
            home = row.get('home_score_extra_time', row['home_score'])
            away = row.get('away_score_extra_time', row['away_score'])
            if home > away:
                return row['home_team_name']
            elif away > home:
                return row['away_team_name']
            else:
                home_pen = row.get('home_score_penalties', 0)
                away_pen = row.get('away_score_penalties', 0)
                if home_pen > away_pen:
                    return row['home_team_name']
                elif away_pen > home_pen:
                    return row['away_team_name']
                else:
                    return "Draw"
        else:
            if row['home_score'] > row['away_score']:
                return row['home_team_name']
            elif row['home_score'] < row['away_score']:
                return row['away_team_name']
            else:
                return "Draw"

    df_matches['winner'] = df_matches.apply(get_knockout_winner, axis=1)

    # ---- MATCH TABLE ----
    df_display = df_matches[['match_id', 'stage', 'match_label', 'home_score', 'away_score', 'winner', 'match_date']].copy()
    df_display = df_display.rename(columns={'home_score':'home_goals','away_score':'away_goals','match_date':'date'})
    
    st.subheader("🏟️ Matches in Selected Competition/Season")
    st.dataframe(df_display)
    
    # ---- TEAM STATS ----
    teams = pd.unique(df_matches[['home_team_name','away_team_name']].values.ravel())
    team_stats = []
    for team in teams:
        home = df_matches[df_matches['home_team_name']==team]
        away = df_matches[df_matches['away_team_name']==team]
        matches_played = len(home)+len(away)
        wins = sum(home['winner'] == team) + sum(away['winner'] == team)
        draws = sum(home['winner'] == "Draw") + sum(away['winner'] == "Draw")
        losses = matches_played-wins-draws 
        goals_scored = home['home_score'].sum() + away['away_score'].sum()
        goals_conceded = home['away_score'].sum() + away['home_score'].sum()
        goals_diff =  goals_scored - goals_conceded
        team_stats.append({
            'team': team,
            'matches_played': matches_played,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goals_diff
        })
    df_team_stats = pd.DataFrame(team_stats).sort_values('matches_played', ascending=False)
    st.subheader("📊 Team Level Summary Stats")
    st.dataframe(df_team_stats)
    
    # ---- KNOCKOUT PROGRESSION ----
    st.subheader("🏆 Knockout Stage Progression")
    knockout_stages = ['Round of 16','Quarter-finals','Semi-finals','Final']
    df_knockouts = df_display[df_display['stage'].isin(knockout_stages)].copy()
    stage_order = {stage: i for i, stage in enumerate(knockout_stages)}
    df_knockouts['stage_order'] = df_knockouts['stage'].map(stage_order)
    df_knockouts = df_knockouts.sort_values(['stage_order','date'])
    for stage in knockout_stages:
        stage_matches = df_knockouts[df_knockouts['stage']==stage]
        if not stage_matches.empty:
            with st.expander(f"{stage}"):
                for _, row in stage_matches.iterrows():
                    st.write(f"{row['match_label']} → Winner: {row['winner']}")

# ==========================
# ---- LOAD EVENTS ---------
# ==========================
if "comp_id" in st.session_state and "season_id" in st.session_state:
    st.subheader("📋 Match Events and Team/Player Stats")
    match_id = st.selectbox("Select a Match", df_display['match_id'].unique())
    url_events = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
    events = requests.get(url_events).json()
    df_events = pd.DataFrame(events)
    
    # Clean events
    df_events['team_name'] = df_events['team'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
    df_events['player_name'] = df_events['player'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
    df_events['event_type'] = df_events['type'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
    df_events['minute'] = df_events['minute']
    df_events['second'] = df_events['second']
    df_events['location_x'] = df_events['location'].apply(lambda x: x[0] if isinstance(x, list) else np.nan)
    df_events['location_y'] = df_events['location'].apply(lambda x: x[1] if isinstance(x, list) else np.nan)
    df_events_clean = df_events[['id','team_name','player_name','event_type','minute','second','location','location_x','location_y']]

    st.subheader("First 20 Cleaned Events")
    st.dataframe(df_events_clean.head(20))
    
    # ---- TEAM EVENTS ----
    team_name = st.selectbox("Select a Team", df_events_clean['team_name'].dropna().unique())
    df_team_events = df_events_clean[df_events_clean['team_name'] == team_name]
    st.write(f"Events for {team_name}")
    st.dataframe(df_team_events)
    
    total_passes = len(df_team_events[df_team_events['event_type']=='Pass'])
    df_team_shots = df_team_events[df_team_events['event_type']=='Shot']
    total_shots = len(df_team_shots)
    possession_percent = round((len(df_team_events)/len(df_events_clean))*100,2)
    
    st.subheader(f"Summary for {team_name}")
    st.write(f"**Total Passes:** {total_passes}")
    st.write(f"**Total Shots:** {total_shots}")
    st.write(f"**Possession (approximate):** {possession_percent}%")
    
    # ---- PLAYER CONTRIBUTIONS ----
    key_events = ['Pass','Shot','Dribble','Foul Committed','Foul Won']
    df_player_summary = (
        df_events[df_events['team_name']==team_name]
        .query("event_type in @key_events")
        .groupby(['player_name','event_type'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=key_events, fill_value=0)
        .reset_index()
    )
    df_player_summary['Total'] = df_player_summary[key_events].sum(axis=1)
    df_player_summary = df_player_summary.sort_values('Total', ascending=False)
    st.subheader("Player Contributions (Key Events)")
    st.dataframe(df_player_summary.head(10))
    
    # ---- PLAYER HEATMAP ----
    st.subheader("Player Heatmap")
    selected_player = st.selectbox("Select Player", df_events['player_name'].dropna().unique())
    
    player_events = df_events[df_events['player_name']==selected_player]
    x = player_events['location_x'].dropna()
    y = player_events['location_y'].dropna()
    
    def draw_pitch(ax):
        ax.set_xlim(0,120)
        ax.set_ylim(0,80)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('green')
        for spine in ax.spines.values():
            spine.set_color('white')
        return ax
    
    if len(x)>0:
        fig, ax = plt.subplots(figsize=(8,6))
        draw_pitch(ax)
        sns.kdeplot(x=x, y=y, fill=True, alpha=0.7, levels=50, cmap='Reds')
        ax.set_title(f"Heatmap for {selected_player}")
        st.pyplot(fig)
    else:
        st.info("No location data available for this player.")
    
    # ---- PASSING NETWORK ----
    st.subheader("Team Passing Network")
    df_passes = df_events[df_events['event_type']=='Pass'].copy()
    df_passes['passer'] = df_passes['player_name']
    df_passes['receiver'] = df_passes['pass'].apply(lambda x: x.get('recipient', {}).get('name') if isinstance(x, dict) and x.get('recipient') else None)
    df_passes['x'] = df_passes['location_x']
    df_passes['y'] = df_passes['location_y']
    
    selected_team = st.selectbox("Select Team for Passing Network", df_passes['team_name'].dropna().unique())
    df_team_passes = df_passes[df_passes['team_name']==selected_team]
    
    G = nx.DiGraph()
    for _, row in df_team_passes.dropna(subset=['passer','receiver']).iterrows():
        if G.has_edge(row['passer'], row['receiver']):
            G[row['passer']][row['receiver']]['weight'] += 1
        else:
            G.add_edge(row['passer'], row['receiver'], weight=1)
    
    player_positions = df_team_passes.groupby('passer')[['x','y']].mean().reset_index()
    pos = {row['passer']:(row['x'], 80-row['y']) for _,row in player_positions.iterrows()}
    
    fig, ax = plt.subplots(figsize=(10,7))
    draw_pitch(ax)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(G,pos,arrowstyle='->',edge_color='red',width=[d['weight']*0.5 for _,_,d in G.edges(data=True)],ax=ax)
    ax.set_title(f"{selected_team} Passing Network")
    st.pyplot(fig)
