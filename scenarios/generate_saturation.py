import random

def generate_saturation():
    threats = []
    tid = 1

    # Ballistic
    for _ in range(20):
        threats.append({
            "id": f"T{tid}",
            "position": [random.uniform(350,450), random.uniform(-30,30), random.uniform(100,140)],
            "velocity": [random.uniform(-4.8,-3.8), random.uniform(-0.2,0.2), random.uniform(-0.4,-0.2)],
            "rcs": "LARGE"
        })
        tid += 1

    # Cruise
    for _ in range(40):
        threats.append({
            "id": f"T{tid}",
            "position": [random.uniform(80,110), random.uniform(-20,20), random.uniform(10,20)],
            "velocity": [random.uniform(-1.2,-0.9), random.uniform(-0.1,0.1), 0],
            "rcs": "MEDIUM"
        })
        tid += 1

    # Rockets
    for _ in range(400):
        threats.append({
            "id": f"T{tid}",
            "position": [random.uniform(60,80), random.uniform(-10,10), random.uniform(5,12)],
            "velocity": [random.uniform(-1.0,-0.8), random.uniform(-0.05,0.05), 0],
            "rcs": "SMALL"
        })
        tid += 1

    # Drones
    for _ in range(60):
        threats.append({
            "id": f"T{tid}",
            "position": [random.uniform(45,60), random.uniform(-10,10), random.uniform(0.5,2.0)],
            "velocity": [random.uniform(-0.25,-0.15), random.uniform(-0.02,0.02), 0],
            "rcs": "MICRO"
        })
        tid += 1

    return {
        "name": "scenario_saturation",
        "description": "Full coordinated saturation attack. All layers activate.",
        "threats": threats
    }