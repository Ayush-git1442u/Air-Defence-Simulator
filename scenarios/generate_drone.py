import random

def generate_drone_only(n=600):
    return {
        "name": "scenario_drone_only",
        "description": f"{n} drones approaching from the east. Only Akash should activate.",
        "threats": [
            {
                "id": f"D{i+1}",
                "position": [
                    random.uniform(45, 60),
                    random.uniform(-10, 10),
                    random.uniform(0.5, 2.0)
                ],
                "velocity": [
                    random.uniform(-0.25, -0.15),
                    random.uniform(-0.02, 0.02),
                    0
                ],
                "rcs": "MICRO"
            }
            for i in range(n)
        ]
    }