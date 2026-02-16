# ============================================================
# Graph Agent Multi-Model Experiment Runner
# ============================================================

import networkx as nx
import requests
import json
import re
import pickle
import time
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = "directed_test.gpickle"
QUERY_FILE = "directed_test_query.txt"
OUTPUT_FILE = "experiment_output.txt"

# Provide 3 OpenRouter keys here
OPENROUTER_KEYS = [
    
]

MODELS = [

    # ------------------------
    # OpenAI (Very Strong Tool Discipline)
    # ------------------------
    "openai/gpt-4o",
    "openai/gpt-4o-mini",

    # ------------------------
    # Anthropic (Excellent Structured Output)
    # ------------------------
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",

    # ------------------------
    # Moonshot (Strong Reasoning)
    # ------------------------
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k1.5-128k",

    # ------------------------
    # Meta Llama (Good Open Models)
    # ------------------------
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",

    # ------------------------
    # Google Gemini (Strong Reasoning)
    # ------------------------
    "google/gemini-1.5-pro",

    # ------------------------
    # Mistral Large (Decent Structured Behavior)
    # ------------------------
    "mistralai/mistral-large"
]


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TEMPERATURE = 0.0
WORKING_MEMORY_SIZE = 15
MAX_TOOL_CALLS = 50

# ============================================================
# KEY ROTATION HANDLER
# ============================================================

class OpenRouterClient:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.keys[self.index]}",
            "Content-Type": "application/json"
        }

    def rotate_key(self):
        self.index = (self.index + 1) % len(self.keys)
        time.sleep(1)


client = OpenRouterClient(OPENROUTER_KEYS)

# ============================================================
# LLM CALL
# ============================================================

def call_llm(messages, model):

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "\n".join(messages)}
        ],
        "temperature": TEMPERATURE
    }

    for attempt in range(len(OPENROUTER_KEYS)):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=client.get_headers(),
                json=payload,
                timeout=60
            )

            if response.status_code == 429 or response.status_code == 401:
                client.rotate_key()
                continue

            result = response.json()

            if "choices" in result:
                return {"response": result["choices"][0]["message"]["content"]}

        except Exception:
            client.rotate_key()
            continue

    return {"response": ""}

# ============================================================
# QUERY LOADER
# ============================================================

def load_queries(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        query = delete_edge = answer = None

        for line in lines:
            if line.startswith("QUERY:"):
                query = line.replace("QUERY:", "").strip()
            elif line.startswith("DELETE_EDGE:"):
                delete_edge = line.replace("DELETE_EDGE:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()

        if query and delete_edge and answer:
            h, r, t = [x.strip() for x in delete_edge.split("|")]
            queries.append({
                "query": query,
                "subject": h,
                "relation": r,
                "object": t,
                "answer": answer
            })

    return queries

# ============================================================
# MEMORY
# ============================================================

class WorkingMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add(self, item):
        if item not in self.memory:
            if len(self.memory) >= self.capacity:
                self.memory.pop(0)
            self.memory.append(item)

    def forget(self, item):
        if item in self.memory:
            self.memory.remove(item)

# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(arguments, G, memory, visited_nodes, visited_edges):

    action = arguments.get("action")

    if action == "list_nodes":
        return list(G.nodes)

    elif action == "get_out_neighbors":
        node = arguments.get("node")
        if node not in G:
            return []

        results = []
        for _, target, key in G.out_edges(node, keys=True):
            results.append({"node": target, "relation": key})
            visited_nodes.update([node, target])
            visited_edges.add((node, key, target))
            memory.add(target)

        return results

    elif action == "get_in_neighbors":
        node = arguments.get("node")
        if node not in G:
            return []

        results = []
        for source, _, key in G.in_edges(node, keys=True):
            results.append({"node": source, "relation": key})
            visited_nodes.update([source, node])
            visited_edges.add((source, key, node))
            memory.add(source)

        return results

    return "INVALID_ACTION"

# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(query, G, model):

    memory = WorkingMemory(WORKING_MEMORY_SIZE)
    messages = []
    visited_nodes = set()
    visited_edges = set()

    system_prompt = """
You are a graph reasoning agent.

Output STRICT JSON ONLY.

Tool call:
{"action":"get_out_neighbors","node":"NAME"}

Final answer:
{"final_answer":"ANSWER"}

No explanations.
"""

    messages.append(system_prompt)
    messages.append(f"Question: {query}")

    tool_calls = 0

    while tool_calls < MAX_TOOL_CALLS:

        response_json = call_llm(messages, model)
        raw = response_json.get("response", "").strip()

        try:
            parsed = json.loads(raw)
        except:
            tool_calls += 1
            continue

        if "action" in parsed:
            tool_result = execute_tool(parsed, G, memory, visited_nodes, visited_edges)
            messages.append(f"Tool result: {json.dumps(tool_result)}")
            tool_calls += 1
            continue

        if "final_answer" in parsed:
            return parsed["final_answer"], visited_nodes, visited_edges

        tool_calls += 1

    return None, visited_nodes, visited_edges

# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

def main():

    with open(GRAPH_PATH, "rb") as f:
        G_original = pickle.load(f)

    queries = load_queries(QUERY_FILE)

    total_runs = len(queries) * len(MODELS)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:

        with tqdm(total=total_runs, desc="Running Experiment") as pbar:

            for model in MODELS:

                out_file.write(f"\n\n================ MODEL: {model} =================\n")

                for q in queries:

                    G = G_original.copy()

                    subject = q["subject"]
                    relation = q["relation"]
                    obj = q["object"]

                    if G.has_edge(subject, obj, key=relation):
                        G.remove_edge(subject, obj, key=relation)

                    answer, visited_nodes, visited_edges = run_agent(
                        q["query"], G, model
                    )

                    result = "YES" if answer and re.search(
                        re.escape(q["answer"]), str(answer), re.IGNORECASE
                    ) else "NO"

                    out_file.write("\n----------------------------------------\n")
                    out_file.write(f"QUERY: {q['query']}\n")
                    out_file.write(f"MODEL: {model}\n")
                    out_file.write(f"FINAL ANSWER: {answer}\n")
                    out_file.write(f"EXPECTED: {q['answer']}\n")
                    out_file.write(f"RESULT: {result}\n")
                    out_file.write(f"VISITED NODES: {sorted(list(visited_nodes))}\n")
                    out_file.write(f"VISITED EDGES: {list(visited_edges)}\n")

                    pbar.update(1)

    print("\nExperiment complete. Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
