# 🛡️Multilayer Air Defense Simulation

A personal hobby project simulating India's multilayer air defense network — S-400, Barak-8, and Akash — responding to coordinated multi-type aerial attacks in real time.

Built purely out of curiosity and interest in defense systems and simulation. No business angle, no commercial intent — just a passion project.

---

## What It Does

Simulates a coordinated air attack against a defended origin point. Threats come from all directions at different altitudes and speeds, and the three AD layers respond in a realistic, staggered fashion — each layer engaging the threats it's best suited for.

The simulation runs in Python, exports JSON, and replays in a 3D Three.js browser visualization.

---

## Threat Types

| Type | Range | Altitude | Speed | Behavior |
|------|-------|----------|-------|----------|
| Ballistic Missile | 300–400 km | 90–140 km | ~4 km/s | High arc, gravity descent |
| Cruise Missile | 80–110 km | 8–20 km | ~1 km/s | Terrain hugging, zigzag |
| Rocket | 55–80 km | 5–12 km | ~0.8 km/s | Straight or jinking |
| Drone | 45–60 km | 0.5–2 km | ~0.2 km/s | Erratic, low and slow |

### Maneuver Types
- **STRAIGHT** — direct approach
- **ZIGZAG** — lateral oscillation
- **CORKSCREW** — spiral descent
- **JINK** — random direction changes
- **TERRAIN_HUG** — drops altitude as it approaches

---

## AD Layer Specs

| System | Range | Altitude | Ammo | Reaction Time | Best Against |
|--------|-------|----------|------|---------------|--------------|
| S-400 | 400 km | 10–185 km | 16 | 9s | Ballistic, Cruise |
| Barak-8 | 100 km | 0.02–16 km | 32 | 5s | Cruise, Rocket, Drone |
| Akash | 60 km | 0.02–18 km | 24 | 8s | Drone, Rocket, Cruise |

Each layer has its own physical launcher model in the 3D view, positioned at different spots on the battlefield.

---

## File Structure

```
AirDefSim/
├── core/
│   ├── classifier.py        # rule-based threat classifier
│   ├── router.py            # minimum-sufficiency AD router
│   └── simulation.py        # main tick loop, staggered launches, re-engagement
├── entities/
│   ├── threat.py            # Threat class + ManeuverType enum
│   ├── ad_layer.py          # ADLayer class + India layers factory
│   └── interceptor.py       # proportional navigation
├── scenarios/
│   ├── generate_ballistic.py
│   ├── generate_drone.py
│   ├── generate_saturation.py
│   └── wave_generator.py    # multidirectional spawning + maneuver assignment
├── ai/
│   ├── rl_environment.py    # Gymnasium wrapper (AirDefenseEnv)
│   ├── train.py             # PPO training
│   ├── evaluate.py          # RL vs hardcoded comparison
│   └── models/              # saved model checkpoints
├── tests/
│   └── manual_scenario.py   # custom scenario runner
├── output/                  # JSON exports go here
├── visualization/
│   └── air_defense_sim_3d_enhanced.html   # 3D Three.js viewer
├── main.py
└── requirements.txt
```

---

## How to Run

**Install dependencies:**
```bash
pip install numpy scikit-learn gymnasium stable-baselines3 torch
```

**Run a manual scenario:**
```bash
python main.py manual --ballistic 5 --cruise 10 --rocket 8 --drone 15
```

**Run with hardcoded router (no RL model needed):**
```bash
python main.py manual --hardcoded
```

**Train the RL agent:**
```bash
python main.py train --steps 2000000 --difficulty random --envs 8
```

**Evaluate RL vs hardcoded:**
```bash
python main.py evaluate --episodes 100 --difficulty hard
```

**View the simulation:**
1. Run any scenario — it exports to `output/`
2. Open `visualization/air_defense_sim_3d_enhanced.html` in a browser
3. Use the threat count controls in the top bar to regenerate scenarios on the fly

---

## 3D Visualization Features

- **Separate launcher models** — S-400 (center), Barak-8 (right flank), Akash (left flank)
- **Distinct projectile shapes** — ballistic missiles, cruise missiles, rockets, X-wing drones
- **Correct orientation** — nose always points in direction of travel
- **Wave timing** — drones first, then rockets, cruise, ballistics last
- **Maneuver trails** — color-coded by maneuver type
- **Live regeneration** — change threat counts in the UI and relaunch instantly
- **Staggered AD response** — Barak-8 fires at t+5s, Akash at t+8s, S-400 at t+9s

---

## AI / RL Component

An RL agent (PPO via Stable-Baselines3) was trained as an alternative to the hardcoded router. Current results:

| Router | Success Rate | Notes |
|--------|-------------|-------|
| Hardcoded | 74.2% | Rule-based minimum sufficiency |
| RL Agent (2M steps) | 73.5% | Needs curriculum training |

The RL agent was trained on an i5-13420H. 2M steps took ~45 minutes. Next step is curriculum training (easy → medium → hard).

---

## Key Design Decisions

- **2D horizontal range checks** — `distance_to_origin()` excludes altitude, matching real-world radar range measurement
- **Re-engagement queue** — missed threats go back in queue and are retried every tick
- **Staggered launches** — each AD layer respects its `reaction_time` before engaging
- **Multidirectional spawning** — threats spawn at random bearing (0–360°), always heading toward origin

---

## Last Simulation Results

120 threats | 34.2% success rate | S-400 and Barak-8 fully depleted

---

## Tech Stack

- **Python 3.10+** — simulation engine
- **NumPy** — physics and vector math
- **Gymnasium + Stable-Baselines3** — RL environment and PPO training
- **Three.js r128** — 3D browser visualization
- **HTML/CSS/JS** — visualization UI

---

*Built for fun. No commercial use.*
