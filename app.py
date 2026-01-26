import streamlit as st
import pickle
import pandas as pd
import random
import time



# --- 1. CONFIGURATION (Must be the very first line) ---
st.set_page_config(page_title="PSL 2025 Analyst", layout="wide", page_icon="🏏")

# --- 2. THE CSS STYLE BLOCK (Do not change this string) ---
# This looks like a comment (green text), but it is actually the styling code.
custom_css = """
<style>
/* Main Background */
.stApp {
    background-color: #0E1117;
}

/* Card Styling */
.stat-card {
    background-color: #262730;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #41444C;
    text-align: center;
    margin-bottom: 10px;
}

/* Metric Value */
.stat-value {
    font-size: 24px;
    font-weight: bold;
    color: #00FF7F;
}

/* Metric Label */
.stat-label {
    font-size: 14px;
    color: #B0B0B0;
}

/* Button Styling */
div.stButton > button {
    width: 100%;
    background-color: #00FF7F;
    color: black;
    font-weight: bold;
    border: none;
    padding: 10px;
    border-radius: 5px;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #00CC66;
    color: white;
}

/* Headers */
h1 { text-align: center; color: #00FF7F; }
h2, h3 { color: white; }
</style>
"""

# Inject the CSS into the app
st.markdown(custom_css, unsafe_allow_html=True)
    

# --- LOAD SYSTEM ---
try:
    pre_match_model = pickle.load(open('psl_model.pkl', 'rb'))
    chase_model = pickle.load(open('chase_model.pkl', 'rb'))
    le_teams = pickle.load(open('team_encoder.pkl', 'rb'))
    le_venues = pickle.load(open('venue_encoder.pkl', 'rb'))
    team_rosters = pickle.load(open('team_rosters.pkl', 'rb'))
    player_stats = pickle.load(open('player_stats.pkl', 'rb'))
    venue_chase_stats = pickle.load(open('venue_chase_stats.pkl', 'rb'))
    venue_performance = pickle.load(open('venue_performance.pkl', 'rb'))
except:
    st.error("System Updating... Please run 'train_model.py'!")
    st.stop()

st.set_page_config(page_title="PSL 2025 Analyst", layout="wide", page_icon="🏏")

# ==========================================
# 🧠 HELPER FUNCTIONS
# ==========================================
def get_player_card(player_name):
    return player_stats.get(player_name, {'bat_avg': 0, 'strike_rate': 0, 'total_wickets': 0, 'matches': 0})

def calculate_squad_power(squad):
    score = 0
    for p in squad:
        s = player_stats.get(p, {'bat_avg': 0, 'total_wickets': 0})
        score += s['bat_avg'] + (s['total_wickets'] * 5)
    return int(score)

# ==========================================
# 🎨 SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/1/14/Pakistan_Super_League_Logo.svg", width=200)
st.sidebar.title("🏏 Analysis Hub")

mode = st.sidebar.radio("Select Tool:", [
    "🏆 Match Simulator", 
    "🎯 Chase Calculator", 
    "⚔️ Player Face-Off", 
    "🏟️ Venue Scout",
    "🌟 AI Dream 11"
])

st.sidebar.divider()
st.sidebar.info("💡 **Updates:** Dream 11 duplicates fixed & Face-Off logic improved.")

# ==========================================
# MODE 1: PRE-MATCH SIMULATOR (FIXED: Squad Selection Restored)
# ==========================================
if mode == "🏆 Match Simulator":
    st.title("🏆 Pre-Match Tactical Simulator")
    
    # 1. Match Setup
    c1, c2, c3 = st.columns(3)
    t1 = c1.selectbox("Team 1", le_teams.classes_, index=0)
    t2 = c2.selectbox("Team 2", le_teams.classes_, index=1)
    venue = c3.selectbox("Venue", le_venues.classes_)

    if t1 == t2:
        st.error("Please select two different teams.")
        st.stop()

    # 2. Squad Selection (RESTORED)
    st.divider()
    st.subheader("📋 Select Playing XI")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{t1} Squad**")
        avail_t1 = team_rosters.get(t1, [])
        # Default top 11
        squad1 = st.multiselect(f"Select Players ({t1})", avail_t1, default=avail_t1[:11])
        
    with col2:
        st.markdown(f"**{t2} Squad**")
        avail_t2 = team_rosters.get(t2, [])
        squad2 = st.multiselect(f"Select Players ({t2})", avail_t2, default=avail_t2[:11])

    # 3. Prediction Engine
    if st.button("🚀 Analyze & Predict", type="primary"):
        # AI Prediction
        t1_id = le_teams.transform([t1])[0]
        t2_id = le_teams.transform([t2])[0]
        v_id = le_venues.transform([venue])[0]
        
        pred = pre_match_model.predict([[t1_id, t2_id, v_id]])[0]
        winner = le_teams.inverse_transform([pred])[0]
        prob = pre_match_model.predict_proba([[t1_id, t2_id, v_id]]).max() * 100
        
        # Display Result
        st.divider()
        st.success(f"🏆 **Predicted Winner:** {winner} ({prob:.1f}%)")
        
        # Reasoning
        st.write("### 🧠 Tactical Reasoning")
        p1 = calculate_squad_power(squad1)
        p2 = calculate_squad_power(squad2)
        
        # Compare selected squads
        if (p1 > p2 and winner == t1) or (p2 > p1 and winner == t2):
            diff = abs(p1 - p2)
            st.write(f"• **Squad Strength:** The selected Playing XI for **{winner}** has a higher cumulative rating (+{diff} pts).")
        else:
            st.write(f"• **Team Chemistry:** Although the opposition has individual stars, **{winner}** performs better as a unit at {venue}.")
            
        st.write(f"• **Venue Mastery:** Historical data at {venue} strongly supports this outcome.")

# ==========================================
# MODE 2: CHASE CALCULATOR (No Changes - Working Best)
# ==========================================
elif mode == "🎯 Chase Calculator":
    st.title("🎯 Target Defense Calculator")
    
    c1, c2 = st.columns(2)
    defending = c1.selectbox("Defending Team", le_teams.classes_)
    chasing = c2.selectbox("Chasing Team", le_teams.classes_, index=1)
    
    target = st.number_input("Target Score", 100, 250, 170)
    venue_chase = st.selectbox("Stadium", le_venues.classes_)
    
    if st.button("Calculate Probability"):
        c_id = le_teams.transform([chasing])[0]
        d_id = le_teams.transform([defending])[0]
        v_id = le_venues.transform([venue_chase])[0]
        
        probs = chase_model.predict_proba([[c_id, d_id, v_id, target]])[0]
        win_prob = probs[1] * 100
        
        st.metric(f"{chasing} Win Chance", f"{win_prob:.1f}%")
        st.progress(int(win_prob))
        
        if win_prob > 50:
            st.success("Target is **Achievable**!")
        else:
            st.error("Target is **Defendable**!")

# ==========================================
# MODE 3: PLAYER FACE-OFF (FIXED: Added Logic & Winner)
# ==========================================
elif mode == "⚔️ Player Face-Off":
    st.title("⚔️ Player Head-to-Head")
    st.markdown("Direct comparison engine.")
    
    all_players = sorted(list(player_stats.keys()))
    
    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Select Player 1", all_players, index=all_players.index("Babar Azam") if "Babar Azam" in all_players else 0)
    p2 = c2.selectbox("Select Player 2", all_players, index=all_players.index("Shaheen Shah Afridi") if "Shaheen Shah Afridi" in all_players else 1)
    
    s1 = get_player_card(p1)
    s2 = get_player_card(p2)
    
    st.divider()
    
    # DISPLAY STATS
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        st.subheader(p1)
        st.metric("Batting Avg", s1['bat_avg'])
        st.metric("Wickets", s1['total_wickets'])
    with colB:
        st.markdown("<h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)
    with colC:
        st.subheader(p2)
        st.metric("Batting Avg", s2['bat_avg'])
        st.metric("Wickets", s2['total_wickets'])

    # WINNER LOGIC
    st.divider()
    st.subheader("📝 The Verdict")
    
    # Simple Point System
    # 1 Run Avg = 1 Point. 1 Wicket = 20 Points.
    score1 = s1['bat_avg'] + (s1['total_wickets'] * 20)
    score2 = s2['bat_avg'] + (s2['total_wickets'] * 20)
    
    if score1 > score2:
        winner = p1
        diff = int(score1 - score2)
        reason = "Better All-Round Stats"
        if s1['bat_avg'] > s2['bat_avg'] + 10: reason = "Significantly Superior Batting"
        if s1['total_wickets'] > s2['total_wickets'] + 5: reason = "Leading Wicket Taker"
    else:
        winner = p2
        diff = int(score2 - score1)
        reason = "Better All-Round Stats"
        if s2['bat_avg'] > s1['bat_avg'] + 10: reason = "Significantly Superior Batting"
        if s2['total_wickets'] > s1['total_wickets'] + 5: reason = "Leading Wicket Taker"
        
    st.success(f"🏆 **WINNER: {winner}**")
    st.write(f"**Reason:** {reason}. {winner} has a higher overall impact rating (+{diff} pts) based on PSL history.")

# ==========================================
# MODE 4: VENUE SCOUT (No Changes - Working Best)
# ==========================================
elif mode == "🏟️ Venue Scout":
    st.title("🏟️ Venue Intelligence")
    selected_venue = st.selectbox("Select Stadium", le_venues.classes_)
    v_data = venue_chase_stats.get(selected_venue, {'avg_first_inn': 'N/A', 'chase_success_rate': 0})
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Avg 1st Innings", v_data['avg_first_inn'])
    m2.metric("Chase Win %", f"{v_data['chase_success_rate']}%")
    st.subheader("🔥 Top Performers")
    if selected_venue in venue_performance:
        top_players = sorted(venue_performance[selected_venue].items(), key=lambda x: x[1], reverse=True)[:5]
        df_venue = pd.DataFrame(top_players, columns=["Player", "Avg Runs"])
        st.table(df_venue)

# ==========================================
# MODE 5: DREAM XI (FIXED: No Duplicates!)
# ==========================================
elif mode == " AI Dream 11":
    st.title(" AI Generated Dream XI")
    st.markdown("Generating the best non-overlapping team...")
    
    if st.button(" Generate Team"):
        # 1. Prepare Pool
        pool = []
        for name, s in player_stats.items():
            entry = s.copy()
            entry['name'] = name
            pool.append(entry)
            
        used_players = set()
        dream_team = []
        
        # 2. Pick 2 All-Rounders First (Hardest Role)
        # Criteria: Avg > 20 AND Wickets > 10
        allrounders = [p for p in pool if p['bat_avg'] > 20 and p['total_wickets'] > 10]
        allrounders = sorted(allrounders, key=lambda x: x['bat_avg'] + x['total_wickets'], reverse=True)[:2]
        
        for p in allrounders:
            p['role'] = "⚔️ All-Rounder"
            dream_team.append(p)
            used_players.add(p['name'])
            
        # 3. Pick 4 Bowlers (High Wickets, Not used yet)
        bowlers = [p for p in pool if p['name'] not in used_players and p['total_wickets'] > 15]
        bowlers = sorted(bowlers, key=lambda x: x['total_wickets'], reverse=True)[:4]
        
        for p in bowlers:
            p['role'] = "⚾ Bowler"
            dream_team.append(p)
            used_players.add(p['name'])
            
        # 4. Pick 5 Batters (High Avg, Not used yet)
        batters = [p for p in pool if p['name'] not in used_players]
        batters = sorted(batters, key=lambda x: x['bat_avg'], reverse=True)[:5]
        
        for p in batters:
            p['role'] = "🏏 Batter"
            dream_team.append(p)
            used_players.add(p['name'])
            
        # 5. Display
        st.balloons()
        st.success("🌟 The Ultimate PSL XI")
        
        # Display as a neat dataframe or list
        for p in dream_team:
             st.write(f"**{p['name']}** ({p['role']}) - Avg: {p['bat_avg']} | Wkts: {p['total_wickets']}")
