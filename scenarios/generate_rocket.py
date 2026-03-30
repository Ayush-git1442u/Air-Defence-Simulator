import random

def generate_rocket_only(n=100):
    return {
        "name": "scenario_rocket_only",
        "description": f"{n} rockets approaching. Barak-8 or Akash should activate.",
        "threats": [
            {
                "id": f"R{i+1}",
                "position": [
                    random.uniform(70, 100),
                    random.uniform(-20, 20),
                    random.uniform(5.0, 15.0)
                ],
                "velocity": [
                    random.uniform(-0.9, -0.8),
                    random.uniform(-0.1, 0.1),
                    0
                ],
                "rcs": "SMALL"
            }
            for i in range(n)
        ]
    }
