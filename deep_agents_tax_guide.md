# LangChain Deep Agents: A Tax Example (Basic to Advanced)

A short guide that uses **two tools** to build a simple Indian income-tax agent, step by step.

Reference: [Deep Agents Quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart)

> The tax rules are simplified figures for FY 2025-26, for learning only. This is not tax advice.

---

## What is a Deep Agent?

`create_deep_agent()` gives you an agent that already has:

- **Planning** (`write_todos`): it makes a to-do list for multi-step work.
- **Files** (`read_file`, `write_file`, `ls`, `edit_file`): it can save notes and reports.
- **Subagents** (`task`): it can hand work to a helper agent.

You add your own **tools** and **instructions** on top.

---

## Step 0: Install

```bash
pip install deepagents langchain-anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Step 1: Basic agent (no tools)

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt="You are an Indian income-tax assistant for FY 2025-26.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is Section 80C?"}]}
)
print(result["messages"][-1].content)
```

**Explanation**

- `model`: which LLM to use, written as `"provider:model"`.
- `system_prompt`: your instructions. They are added to Deep Agents' own built-in prompt.
- `invoke`: runs the agent. The last message in the result is the answer.

---

## Step 2: Add two tools

LLMs often get arithmetic wrong, so the calculation goes into Python functions.

```python
from langchain.tools import tool

NEW_SLABS = [(400000, 0), (800000, .05), (1200000, .10), (1600000, .15),
             (2000000, .20), (2400000, .25), (float("inf"), .30)]
OLD_SLABS = [(250000, 0), (500000, .05), (1000000, .20), (float("inf"), .30)]


def _slab_tax(income, slabs):
    tax, lower = 0, 0
    for upper, rate in slabs:
        if income > lower:
            tax += (min(income, upper) - lower) * rate
        lower = upper
    return tax


@tool
def calculate_tax(salary: float, regime: str, deductions: float = 0) -> dict:
    """Calculate income tax in INR for FY 2025-26.

    Args:
        salary: Annual gross salary in INR.
        regime: "new" or "old".
        deductions: 80C/80D etc. Only used in the old regime.
    """
    if regime == "new":
        taxable = max(0, salary - 75000)          # standard deduction
        tax = _slab_tax(taxable, NEW_SLABS)
        tax = 0 if taxable <= 1200000 else min(tax, taxable - 1200000)  # 87A rebate
    else:
        taxable = max(0, salary - 50000 - deductions)
        tax = _slab_tax(taxable, OLD_SLABS)
        tax = 0 if taxable <= 500000 else tax     # 87A rebate
    return {"regime": regime, "taxable_income": taxable,
            "total_tax": round(tax * 1.04)}       # + 4% cess


@tool
def compare_regimes(salary: float, deductions: float = 0) -> dict:
    """Compare old vs new regime and return the cheaper one."""
    new = calculate_tax.invoke({"salary": salary, "regime": "new"})
    old = calculate_tax.invoke({"salary": salary, "regime": "old",
                                "deductions": deductions})
    best = "new" if new["total_tax"] <= old["total_tax"] else "old"
    return {"new": new, "old": old, "recommended": best}


agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[calculate_tax, compare_regimes],
    system_prompt="You are a tax assistant. Always use the tools for numbers.",
)

result = agent.invoke({"messages": [{"role": "user",
    "content": "Salary 18 lakh, deductions 2 lakh. Which regime is better?"}]})
print(result["messages"][-1].content)
```

**Explanation**

- `@tool` turns a function into a tool. The model reads the **name**, the **type hints** and the **docstring** to decide when and how to call it, so write clear docstrings.
- `tools=[...]`: these are added alongside the built-in tools.
- For this question the agent calls `compare_regimes(1800000, 200000)`. The new regime comes to ₹1,50,800 and the old regime to ₹2,88,600, so it recommends the new regime.

---

## Step 3: Planning and files (built in)

You only need to ask for them in the prompt. No new code is required.

```python
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[calculate_tax, compare_regimes],
    system_prompt=(
        "You are a tax assistant. First make a plan with write_todos. "
        "Save the final computation to /report.md."
    ),
)

result = agent.invoke({"messages": [{"role": "user",
    "content": "Prepare a tax report for salary 15 lakh, deductions 1.5 lakh."}]})

print(result["todos"])                   # the plan the agent made
print(list(result["files"].keys()))      # ['/report.md']
```

**Explanation**

- By default, files are stored in the agent's state (`result["files"]`), not on your disk. This is safe.
- To write real files, add `backend=FilesystemBackend(root_dir="./out", virtual_mode=True)`. Import it from `deepagents.backends`.

---

## Step 4: Subagent

A helper agent with its own prompt and tool.

```python
regime_advisor = {
    "name": "regime-advisor",
    "description": "Compares old vs new tax regime for a salary.",
    "system_prompt": "Use compare_regimes and return a short recommendation.",
    "tools": [compare_regimes],
}

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[calculate_tax],
    subagents=[regime_advisor],
    system_prompt="You are a lead tax consultant. Delegate regime questions to regime-advisor.",
)
```

**Explanation**

- The main agent reads `description` to decide when to delegate. It then calls the built-in `task` tool.
- The subagent works in its own clean context and returns only its final answer. This keeps the main agent's context small.

---

## Step 5: Human approval (advanced)

Pause before a sensitive tool runs, so a person can approve it.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[calculate_tax, compare_regimes],
    interrupt_on={"calculate_tax": {"allowed_decisions": ["approve", "reject"]}},
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "client-1"}}
result = agent.invoke({"messages": [{"role": "user",
    "content": "Tax on 12 lakh, new regime?"}]}, config=config)

if result.get("__interrupt__"):
    action = result["__interrupt__"][0].value["action_requests"][0]
    print("Approve?", action["name"], action["args"])
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}),
                          config=config)

print(result["messages"][-1].content)
```

**Explanation**

- `interrupt_on`: the agent pauses before calling `calculate_tax`. In a real app you would use this for actions like filing a return or making a payment.
- `checkpointer` is **required**. It saves the paused state.
- `Command(resume=...)` continues the run. You must pass the **same `thread_id`**.

---

## Step 6: Long-term memory (advanced)

Remember client details across conversations.

```python
from langgraph.store.memory import InMemoryStore
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[calculate_tax, compare_regimes],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=lambda _rt: ("clients",))},
    ),
    store=InMemoryStore(),
    system_prompt="Save client facts to /memories/client.md and read it at the start.",
)
```

**Explanation**

- Files under `/memories/` go to the **store**, which persists across threads. Other files exist only for the current conversation.
- In production, replace `InMemoryStore` with a database-backed store.

---

## Summary

| Step | Feature | Key parameter |
|---|---|---|
| 1 | Basic agent | `model`, `system_prompt` |
| 2 | Custom tools | `tools` |
| 3 | Planning and files | built in (`write_todos`, `write_file`) |
| 4 | Subagent | `subagents` |
| 5 | Human approval | `interrupt_on` + `checkpointer` |
| 6 | Memory | `backend` + `store` |
