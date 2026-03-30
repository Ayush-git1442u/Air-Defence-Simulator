import random

def generate_cruise_only(n=50):
    return {
        "name": "scenario_cruise_only",
        "description": f"{n} cruise missiles approaching. Barak-8 should activate.",
        "threats": [
            {
                "id": f"C{i+1}",
                "position": [
                    random.uniform(90, 120),
                    random.uniform(-30, 30),
                    random.uniform(10.0, 20.0)
                ],
                "velocity": [
                    random.uniform(-1.1, -0.9),
                    random.uniform(-0.1, 0.1),
                    0
                ],
                "rcs": "MEDIUM"
            }
            for i in range(n)
        ]
    }
