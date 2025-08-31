# generate_data.py
import pandas as pd
import numpy as np

def generate_student_data(n=1000, random_seed=42):
    np.random.seed(random_seed)
    hours_studied = np.round(np.clip(np.random.normal(5, 2, n), 0, 12), 2)  # 0-12 hours
    attendance = np.round(np.clip(np.random.normal(85, 10, n), 50, 100), 1)  # %
    assignments_submitted = np.random.randint(0, 11, n)  # 0-10
    prev_score = np.round(np.clip(np.random.normal(60, 15, n), 0, 100), 1)

    # A synthetic target formula (with noise)
    score = (0.6 * hours_studied * 5) + (0.2 * attendance) + (1.5 * assignments_submitted) + (0.3 * prev_score)
    score += np.random.normal(0, 6, n)  # noise
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        "hours_studied": hours_studied,
        "attendance_pct": attendance,
        "assignments_submitted": assignments_submitted,
        "previous_score": prev_score,
        "final_score": np.round(score, 1)
    })
    return df

if __name__ == "__main__":
    df = generate_student_data(1000)
    df.to_csv("data/student_scores.csv", index=False)
    print("Generated data/student_scores.csv (1000 rows).")
