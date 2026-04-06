# CODECRACK

CodeCrack — AI Code Review Dashboard

title: CodeCrack
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false


AI agent training environment for automated code review with an interactive dashboard, structured difficulty levels, and carefully designed safe-code distractors that evaluate true understanding rather than simple pattern matching.

Key Features

Safe-Code Distractors: Tasks include intentionally safe patterns that may appear vulnerable but are correctly implemented (such as parameterized SQL queries or proper lock handling). This ensures that solutions rely on reasoning instead of superficial detection.

Hybrid Baseline: Combines rule-based detection for common issues with a language model fallback for handling complex or ambiguous scenarios.

Flexible Grading: A ±2 line tolerance ensures that minor positional differences do not unfairly penalize results.

Progressive Difficulty: Tasks are structured across three levels—Easy (1 issue), Medium (2 issues), and Hard (3 issues).

Baseline Performance
[EASY  ] easy_sql_injection      : 1.000  (2 steps, 0 API calls)
[MEDIUM] medium_race_condition   : 1.000  (3 steps, 0 API calls)
[HARD  ] hard_memory_leak        : 1.000  (4 steps, 0 API calls)

Average Score    : 1.000
Average Steps    : 3.0
Total API Calls  : 0

The hybrid baseline relies on predefined issue mappings for built-in tasks, resulting in zero API calls. The language model is only used when evaluating custom or unseen inputs.

Environment Overview
Property	Value
Domain	Software Engineering / Code Review
Tasks	3 (easy → medium → hard)
Reward range	-2.0 to +6.0
Max steps	50 per episode
Grading	0.5×recall + 0.3×precision + 0.2×severity (±2 line tolerance)

Quick Start
# Install dependencies
pip install -r requirements.txt

# Configure API (choose one)
export GROQ_API_KEY=gsk_...
# OR
export TOGETHER_API_KEY=...

# Launch the dashboard
python app.py
# → c

# Run baseline
python inference.py
pyhton baseline.py

# Validate environment
bash validate.sh

Tasks
Easy — SQL Injection Detection

Code: 38-line authentication module
Issue: 1 critical — unsafe string interpolation in SQL query (line 18)

Distractors:

get_user_by_id() uses parameterized queries (safe)
log_attempt() uses f-string in logging (safe — not a query)
get_users_by_role() uses parameterized queries (safe)
render_welcome() uses formatting only for display (safe)
Medium — Race Condition Analysis

Code: 51-line BankAccount class
Issues: 2 high severity

Read-modify-write race in deposit() (line 16)
Time-of-check to time-of-use issue in withdraw() (line 22)

Distractors:

transfer() follows correct lock ordering (safe)
get_balance() performs read-only access (safe)
get_statement() and freeze() use proper locking (safe)
Hard — Memory Leak and Iterator Issues

Code: 60-line TTL cache manager
Issues: 3 (high + medium + high)

Memory leak due to unmanaged listeners (line 11)
Expired entries not removed in get() (line 27)
Unsafe dictionary modification during iteration in cleanup_expired() (line 42)

Distractors:

get_stats() iterates safely without modification
invalidate_prefix() collects keys before deletion
get_active_values() reads values without mutation
API Reference
OpenEnv Interface
from environment import CodeReviewEnv

env = CodeReviewEnv()
obs = env.reset(task_id="easy_sql_injection")

from models import Action
obs, reward, done, info = env.step(Action(
    action_type="identify_issue",
    issue_type="security",
    line_number=18,
    description="SQL injection via unsafe string construction",
    severity="critical"
))

state = env.state()
REST Endpoints
Method	Path	Description
GET	/	Health check
POST	/reset?task_id=...	Reset environment
POST	/step	Execute action
GET	/state	Get current state
GET	/tasks	List all tasks
Grading Formula
score = 0.5 × recall  +  0.3 × precision  +  0.2 × severity_match
Component	Weight	Description
Recall	50%	Percentage of actual issues identified
Precision	30%	Accuracy of reported issues
Severity	20%	Correct classification of severity

Line tolerance: ±2 lines to handle minor variations

Docker Deployment
docker build -t code-review-env .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=gsk_... \
  code-review-env
Hugging Face Spaces

Already deployed at:
https://huggingface.co/spaces/METAHACK/CodeCrack

To update:

git push hf main
Pre-Submission Checklist
bash validate.sh passes
python inference.py completes within time limits
Docker builds successfully
Hugging Face Space responds correctly
/reset endpoint works as expected
All tasks return valid evaluation scores
Dependencies
pydantic==2.6.0 — Typed models
fastapi==0.109.0 — API framework
uvicorn==0.27.0 — Server runtime
openai==1.12.0 — LLM client
python-dotenv==1.0.0 — Environment configuration
Team

Authors: Madhan J and Sahithya BR
Hackathon: Scaler Meta PyTorch OpenEnv Challenge 2025
