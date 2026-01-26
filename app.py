import streamlit as st
import pickle
import pandas as pd
import random
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="PSL Analyst", layout="wide", page_icon="🏏")

# --- 2. CSS STYLING ---
custom_css = """
<style>
.stApp { background-color: #0E1117; }
.stat-card { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #41444C; text-align: center; margin-bottom: 10px; }
.stat-value { font-size: 22px; font-weight: bold; color: #00FF7F; }
.stat-label { font-size: 12px; color: #B0B0B0; }
div.stButton > button { width: 100%; background-color: #00FF7F; color: black; font-weight: bold; border: none; padding: 10px; border-radius: 5px; transition: 0.3s; }
div.stButton > button:hover { background-color: #00CC66; color: white; }
h1 { text-align: center; color: #00FF7F; }
h2, h3 { color: white; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. LOAD SYSTEM ---
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
    st.error("⚠️ System Files Missing! Please run 'train_model.py' first.")
    st.stop()

# --- HELPER FUNCTIONS ---
def metric_card(label, value, prefix=""):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{prefix}{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def get_player_card(player_name):
    return player_stats.get(player_name, {'bat_avg': 0, 'strike_rate': 0, 'total_wickets': 0, 'matches': 0})

def calculate_squad_power(squad):
    score = 0
    for p in squad:
        s = player_stats.get(p, {'bat_avg': 0, 'total_wickets': 0})
        score += s['bat_avg'] + (s['total_wickets'] * 5)
    return int(score)

# --- SIDEBAR ---
st.sidebar.image("https://www.geo.tv/assets/uploads/updates/2021-12-03/385763_1727813_updates.jpeg", width=150)
st.sidebar.markdown("## 🏏 PSL Analysis Hub")

mode = st.sidebar.radio("", [
    "🏆 Match Simulator", 
    "🎯 Chase Calculator", 
    "⚔️ Player Face-Off", 
    "🏟️ Venue Scout",
    "🌟 AI Dream 11"
])
st.sidebar.divider()
st.sidebar.info("v5.0 | Final Edition")

# ==========================================
# MODE 1: MATCH SIMULATOR
# ==========================================
if mode == "🏆 Match Simulator":
    st.title("🏆 Pre-Match Simulator")
    st.markdown("<p style='text-align: center;'>Tactical Squad Analysis</p>", unsafe_allow_html=True)
    st.divider()

    with st.container():
        c1, c2, c3 = st.columns(3)
        t1 = c1.selectbox("Home Team", le_teams.classes_, index=0)
        t2 = c2.selectbox("Away Team", le_teams.classes_, index=1)
        venue = c3.selectbox("Venue", le_venues.classes_)

    if t1 == t2:
        st.warning("⚠️ Please select different teams.")
        st.stop()

    st.subheader("📋 Squad Selection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**🔹 {t1}**")
        avail_t1 = team_rosters.get(t1, [])
        squad1 = st.multiselect(f"Select XI for {t1}", avail_t1, default=avail_t1[:11], label_visibility="collapsed")
    with col2:
        st.markdown(f"**🔸 {t2}**")
        avail_t2 = team_rosters.get(t2, [])
        squad2 = st.multiselect(f"Select XI for {t2}", avail_t2, default=avail_t2[:11], label_visibility="collapsed")

    st.markdown("---")
    
    if st.button("🚀 PREDICT WINNER"):
        t1_id = le_teams.transform([t1])[0]
        t2_id = le_teams.transform([t2])[0]
        v_id = le_venues.transform([venue])[0]
        pred = pre_match_model.predict([[t1_id, t2_id, v_id]])[0]
        winner = le_teams.inverse_transform([pred])[0]
        prob = pre_match_model.predict_proba([[t1_id, t2_id, v_id]]).max() * 100
        
        st.success(f"🏆 PREDICTED WINNER: {winner} ({prob:.1f}%)")
        
        p1 = calculate_squad_power(squad1)
        p2 = calculate_squad_power(squad2)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card(f"{t1} Power", p1)
        with c2: metric_card("Venue Impact", "High" if prob > 60 else "Low")
        with c3: metric_card(f"{t2} Power", p2)
        
        st.caption("Comparison based on cumulative player impact ratings.")

# ==========================================
# MODE 2: CHASE CALCULATOR (FIXED: Added Reasons!)
# ==========================================
elif mode == "🎯 Chase Calculator":
    st.title("🎯 Target Defense Calculator")
    st.markdown("<p style='text-align: center;'>2nd Innings Probability Engine</p>", unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    defending = c1.selectbox("Defending", le_teams.classes_, index=0)
    chasing = c2.selectbox("Chasing", le_teams.classes_, index=1)
    venue_chase = c3.selectbox("Venue", le_venues.classes_)
    target = c4.number_input("Target", 100, 300, 180)

    st.markdown("###") 
    
    if st.button("📉 CALCULATE PROBABILITY"):
        c_id = le_teams.transform([chasing])[0]
        d_id = le_teams.transform([defending])[0]
        v_id = le_venues.transform([venue_chase])[0]
        
        probs = chase_model.predict_proba([[c_id, d_id, v_id, target]])[0]
        win_prob = probs[1] * 100
        
        # Display Result
        st.markdown(f"<h1 style='text-align: center; color: #00FF7F;'>{win_prob:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Win Probability for {chasing}</p>", unsafe_allow_html=True)
        st.progress(int(win_prob))
        
        # --- NEW REASONING SECTION ---
        st.divider()
        st.subheader("🧠 Tactical Analysis")
        
        # Reason 1: Compare Target to Venue Average
        v_stat = venue_chase_stats.get(venue_chase, {'avg_first_inn': 165})
        avg_score = v_stat['avg_first_inn']
        diff = target - avg_score
        
        if diff > 15:
            st.error(f"🔴 **Difficult Chase:** The target of {target} is significantly higher than the average score at {venue_chase} ({avg_score}).")
        elif diff < -15:
            st.success(f"🟢 **Easy Chase:** The target is well below par ({avg_score}). {chasing} should chase this comfortably.")
        else:
            st.warning(f"🟡 **Fighting Total:** The target is very close to the historical average ({avg_score}). It will go down to the wire.")

        # Reason 2: Historical Bias
        if win_prob > 60:
            st.write(f"• **History:** {chasing} has a strong record chasing totals under {target+10}.")
        else:
             st.write(f"• **Pressure:** {defending} is statistically strong at defending scores over {target-10}.")

# ==========================================
# MODE 3: PLAYER FACE-OFF
# ==========================================
elif mode == "⚔️ Player Face-Off":
    st.title("⚔️ Player Face-Off")
    all_players = sorted(list(player_stats.keys()))
    
    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Player 1", all_players, index=all_players.index("Babar Azam") if "Babar Azam" in all_players else 0)
    p2 = c2.selectbox("Player 2", all_players, index=all_players.index("Shaheen Shah Afridi") if "Shaheen Shah Afridi" in all_players else 1)
    
    s1 = get_player_card(p1)
    s2 = get_player_card(p2)
    
    st.divider()
    col_left, col_mid, col_right = st.columns([2, 0.5, 2])
    
    with col_left:
        st.markdown(f"<h3 style='text-align: center;'>{p1}</h3>", unsafe_allow_html=True)
        metric_card("Batting Avg", s1['bat_avg'])
        metric_card("Wickets", s1['total_wickets'])
    with col_mid:
        st.markdown("<br><br><br><h1 style='text-align: center; color: #444;'>VS</h1>", unsafe_allow_html=True)
    with col_right:
        st.markdown(f"<h3 style='text-align: center;'>{p2}</h3>", unsafe_allow_html=True)
        metric_card("Batting Avg", s2['bat_avg'])
        metric_card("Wickets", s2['total_wickets'])

    st.divider()
    score1 = s1['bat_avg'] + (s1['total_wickets'] * 20)
    score2 = s2['bat_avg'] + (s2['total_wickets'] * 20)
    winner = p1 if score1 > score2 else p2
    st.success(f"🏆 VERDICT: **{winner}** is the statistically superior impact player.")

# ==========================================
# MODE 4: VENUE SCOUT
# ==========================================
elif mode == "🏟️ Venue Scout":
    st.title("🏟️ Venue Intelligence")
    sel_venue = st.selectbox("Select Stadium", le_venues.classes_)
    v_data = venue_chase_stats.get(sel_venue, {'avg_first_inn': 'N/A', 'chase_success_rate': 0})
    st.divider()
    c1, c2 = st.columns(2)
    with c1: metric_card("Avg 1st Innings", v_data['avg_first_inn'])
    with c2: metric_card("Chase Win %", f"{v_data['chase_success_rate']}%")
    st.markdown("### 🔥 Top Performers")
    if sel_venue in venue_performance:
        top = sorted(venue_performance[sel_venue].items(), key=lambda x: x[1], reverse=True)[:5]
        df_v = pd.DataFrame(top, columns=["Player", "Avg Runs"])
        st.dataframe(df_v, use_container_width=True, hide_index=True)

# ==========================================
# MODE 5: DREAM XI (FIXED: More Robust Logic!)
# ==========================================
elif mode == "🌟 AI Dream 11":
    st.title("🌟 AI Dream Team")
    st.markdown("The algorithm selects the highest rated non-overlapping XI.")
    
    if st.button("✨ GENERATE SQUAD"):
        with st.spinner("Scouting Database..."):
            time.sleep(1)
            pool = [{'name': k, **v} for k, v in player_stats.items()]
            used = set()
            team = []
            
            # 1. Pick All Rounders (Criteria relaxed slightly to ensure hits)
            ars = sorted([p for p in pool if p['bat_avg']>15 and p['total_wickets']>5], key=lambda x: x['bat_avg']+x['total_wickets'], reverse=True)[:2]
            for p in ars: 
                p['role']="⚔️ All-Rounder"
                team.append(p)
                used.add(p['name'])
            
            # 2. Pick Bowlers
            bwls = sorted([p for p in pool if p['name'] not in used and p['total_wickets']>10], key=lambda x: x['total_wickets'], reverse=True)[:4]
            for p in bwls: 
                p['role']="⚾ Bowler"
                team.append(p)
                used.add(p['name'])
            
            # 3. Pick Batters (Fill the rest)
            needed = 11 - len(team)
            bats = sorted([p for p in pool if p['name'] not in used], key=lambda x: x['bat_avg'], reverse=True)[:needed]
            for p in bats: 
                p['role']="🏏 Batter"
                team.append(p)
                used.add(p['name'])
            
            st.balloons()
            st.subheader("The Ultimate XI")
            
            # Use columns to create a grid layout
            cols = st.columns(4)
            for i, p in enumerate(team):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="stat-card" style="padding:10px; margin-bottom:10px;">
                        <div style="color:#00FF7F; font-weight:bold;">{p['role']}</div>
                        <div style="font-size:16px;">{p['name']}</div>
                        <div style="font-size:12px; color:#aaa;">Avg: {p['bat_avg']} | Wkts: {p['total_wickets']}</div>
                    </div>
                    """, unsafe_allow_html=True)
