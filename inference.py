
import os
from openai import OpenAI
import traceback

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(prompt: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    try:
        task_name = "demo-task"
        env_name = "sql-openenv"
        step_count = 0
        rewards = []

        print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}")

        # Dummy loop (replace with your env logic)
        done = False

        while not done and step_count < 3:
            step_count += 1

            action = run_inference("Give a simple SQL query")

            reward = 0.0
            done = step_count == 3

            rewards.append(f"{reward:.2f}")

            print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

        print(f"[END] success=true steps={step_count} rewards={','.join(rewards)}")

    except Exception as e:
        print(f"[END] success=false steps=0 rewards= error={str(e)}")
        traceback.print_exc()

if __name__ == "_main_":
    main()