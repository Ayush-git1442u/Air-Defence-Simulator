import random

def generate_ballistic_only(n=250):
    return {
        "name": "scenario_ballistic_only",
        "description": f"{n} ballistic missiles on high altitude trajectories. Only S-400 should activate.",
        "threats": [
            {
                "id": f"B{i+1}",
                "position": [
                    random.uniform(350, 450),
                    random.uniform(-50, 50),
                    random.uniform(90, 130)
                ],
                "velocity": [
                    random.uniform(-4.8, -3.8),
                    random.uniform(-0.2, 0.2),
                    random.uniform(-1.2, -0.2)
                ],
                "rcs": "LARGE"
            }
            for i in range(n)
        ]
    }