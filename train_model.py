import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("🚀 Initializing Dual-Core System...")

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv('psl.csv')
    df.columns = df.columns.str.strip().str.lower()
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Auto-detect columns
batter_col = 'batter'
bowler_col = 'bowler'
runs_col = 'batsman_runs'

# --- 2. PRE-MATCH INTELLIGENCE (For Mode 1) ---
print("⚙️ Building Pre-Match Intelligence...")
player_stats = {} 
team_rosters = {}
venue_performance = {}

# (Keep existing logic for rosters/stats - it was good)
all_teams = pd.concat([df['batting_team'], df['bowling_team']]).unique()
for team in all_teams:
    batters = df[df['batting_team'] == team][batter_col].unique()
    bowlers = df[df['bowling_team'] == team][bowler_col].unique()
    team_rosters[team] = sorted(list(set(list(batters) + list(bowlers))))

all_players = pd.concat([df[batter_col], df[bowler_col]]).unique()
for p in all_players:
    p_data = df[df[batter_col] == p]
    bat_avg = p_data[runs_col].sum() / p_data['match_id'].nunique() if not p_data.empty else 0
    
    b_data = df[df[bowler_col] == p]
    wickets = b_data[b_data['is_wicket'] == True].shape[0] if 'is_wicket' in b_data.columns else 0
    
    player_stats[p] = {'bat_avg': round(bat_avg, 2), 'total_wickets': wickets}

# --- 3. CHASE INTELLIGENCE (For Mode 2) ---
print("🎯 Building Chase Engine...")

# We need to aggregate match totals
match_agg = df.groupby(['match_id', 'inning', 'batting_team', 'bowling_team', 'venue', 'winner'])['total_runs'].sum().reset_index()

# Filter only complete matches (Inning 1 and 2 exists)
matches_1 = match_agg[match_agg['inning'] == 1].rename(columns={'total_runs': 'target_set'})
matches_2 = match_agg[match_agg['inning'] == 2].rename(columns={'total_runs': 'chase_score'})

# Merge to get a single row per match: [Chasing Team, Defending Team, Venue, Target, Result]
chase_data = pd.merge(matches_1[['match_id', 'target_set', 'venue']], 
                      matches_2[['match_id', 'batting_team', 'bowling_team', 'winner']], 
                      on='match_id')

# Adjust Target (First Innings Score + 1)
chase_data['target_set'] = chase_data['target_set'] + 1

# Did the chasing team win?
chase_data['chase_successful'] = (chase_data['winner'] == chase_data['batting_team']).astype(int)

# Venue Chase Stats (For Reasoning)
venue_chase_stats = {}
for v in chase_data['venue'].unique():
    v_data = chase_data[chase_data['venue'] == v]
    avg_target = v_data['target_set'].mean()
    success_rate = v_data['chase_successful'].mean() * 100
    avg_score_bat_1 = v_data['target_set'].mean() - 1
    venue_chase_stats[v] = {
        'avg_first_inn': round(avg_score_bat_1),
        'chase_success_rate': round(success_rate, 1)
    }

# Train Chase Model (Logistic Regression is better for Probability here)
le_teams = LabelEncoder()
le_venues = LabelEncoder()

# Fit encoders on ALL teams/venues
all_teams_list = pd.concat([chase_data['batting_team'], chase_data['bowling_team']]).unique()
le_teams.fit(all_teams_list)
le_venues.fit(chase_data['venue'].astype(str))

X_chase = pd.DataFrame()
X_chase['Chasing_Team'] = le_teams.transform(chase_data['batting_team'])
X_chase['Defending_Team'] = le_teams.transform(chase_data['bowling_team'])
X_chase['Venue'] = le_venues.transform(chase_data['venue'])
X_chase['Target'] = chase_data['target_set']
y_chase = chase_data['chase_successful']

chase_model = LogisticRegression() # Good for "Win Probability"
chase_model.fit(X_chase, y_chase)

# --- 4. PRE-MATCH MODEL TRAIN ---
# (Keeping the old one for Mode 1)
matches = df.drop_duplicates(subset=['match_id']).dropna(subset=['winner', 'venue'])
matches = matches[~matches['winner'].astype(str).str.contains('No result', case=False, na=False)]
X_pre = pd.DataFrame()
X_pre['Team1'] = le_teams.transform(matches['batting_team'])
X_pre['Team2'] = le_teams.transform(matches['bowling_team'])
X_pre['Venue'] = le_venues.transform(matches['venue'].astype(str))
y_pre = le_teams.transform(matches['winner'])

pre_match_model = RandomForestClassifier(n_estimators=100, random_state=42)
pre_match_model.fit(X_pre, y_pre)

# --- 5. SAVE EVERYTHING ---
pickle.dump(pre_match_model, open('psl_model.pkl', 'wb'))
pickle.dump(chase_model, open('chase_model.pkl', 'wb')) # NEW FILE
pickle.dump(le_teams, open('team_encoder.pkl', 'wb'))
pickle.dump(le_venues, open('venue_encoder.pkl', 'wb'))
pickle.dump(team_rosters, open('team_rosters.pkl', 'wb'))
pickle.dump(player_stats, open('player_stats.pkl', 'wb'))
pickle.dump(venue_performance, open('venue_performance.pkl', 'wb'))
pickle.dump(venue_chase_stats, open('venue_chase_stats.pkl', 'wb')) # NEW FILE

print("✅ SUCCESS: Intelligence files saved. You can now run the App!")

# /usr/local/bin/python3 -m streamlit run /Users/apple/Desktop/PSL_AI_Project/app.py
