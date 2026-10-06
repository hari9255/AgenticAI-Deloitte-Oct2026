

import json
import sys
from datetime import date
from typing import Literal, Optional

import pdfplumber
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import PDFPlumberLoader

import os
os.environ["Anthropic_API_Key"] = ""
os.environ["TAVILY_API_KEY"] = ""
MODEL = "anthropic:claude-haiku-4-5"   # any tool-calling model works, e.g. "anthropic:claude-sonnet-4-5"


# =============================================================================
# 1. RESPONSE FORMATS (Pydantic schemas)
# =============================================================================



class LineItem(BaseModel):
    hsn_code: str
    quantity: float
    rate: float = Field(description="Unit price before discount")
    taxable_value: float
    gst_rate_percent: float = Field(description="Total GST rate (for CGST+SGST give the combined rate)")
    


class InvoiceData(BaseModel):
    invoice_number: str
    invoice_date: str = Field(description="Invoice date in ISO format YYYY-MM-DD")
    supply_type: str = Field(description="As printed, e.g. 'Inter-State (IGST)'")
    items: list[LineItem]
    total_taxable_value: float
    total_gst: float = 0.0    
    grand_total: float
   


class GSTSlabCheck(BaseModel):
    sl_no: int
    description: str
    hsn_code: str
    is_correct: bool    
    notes: str = ""


class CheckResult(BaseModel):
    check: str = Field(description="What was checked, e.g. 'Item 3 taxable value'")
    status: Literal["PASS", "FAIL", "WARNING"]
    expected: str
    found_on_invoice: str
    details: str = ""


class ValidationReport(BaseModel):
    invoice_number: str
    gst_slab_checks: list[GSTSlabCheck]
    calculation_checks: list[CheckResult]
    date_checks: list[CheckResult]
    supply_type_check: CheckResult
    overall_status: Literal["VALID", "INVALID", "NEEDS_REVIEW"]
    summary: str


# =============================================================================
# 2. TOOLS
# =============================================================================

# ---- Helper (not a tool): read PDF text and pass it to Agent 1 in the prompt ----
def read_pdf_text(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text(layout=True) for page in pdf.pages)


# ---- Agent 2 tools ----------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Calculate a math expression. Use only numbers and + - * / % ( ).
    Example: '2 * 52000 * (1 - 5/100)'  or  '98800 * 18 / 100'"""
    expr = expression.replace(",", "")            # 1,16,584.00 -> 116584.00
    if not set(expr) <= set("0123456789.+-*/%() "):
        return "Error: only numbers and + - * / % ( ) are allowed"
    try:
        return f"{expression} = {round(eval(expr), 2)}"
    except Exception as e:
        return f"Error: {e}"




tavily_search = TavilySearch(max_results=3, topic="general")


# =============================================================================
# 3. SYSTEM PROMPTS
# =============================================================================

EXTRACTOR_PROMPT = """You are an expert Indian GST invoice data-extraction agent.
You will receive the text of an invoice. Extract EVERY field into the InvoiceData schema.
Rules:
- Numbers: strip ₹ and Indian commas (1,16,584.00 -> 116584.00).
- Dates: convert to ISO YYYY-MM-DD (06-Oct-2026 -> 2026-10-06).
- Extract values EXACTLY as printed — do not correct or recalculate anything.
- Multi-line descriptions must be joined into one string."""

VALIDATOR_PROMPT = """You are a meticulous Indian GST audit agent. You receive invoice data as JSON.
Perform ALL checks below. Never do arithmetic in your head — ALWAYS use the `calculator` tool.

A) GST SLAB CHECK (tavily_search)
   - For each item search e.g. "GST rate HSN <code> <description> India latest".
   - GST rates were rationalised in Sept 2025 (GST 2.0: 0%, 5%, 18%, 40% slabs) — prefer the most
     recent official/reputable sources (cbic-gst.gov.in, gst.gov.in, ClearTax, etc.) and check the
     rate applicable on the INVOICE DATE. Cite the source URL.

B) CALCULATION CHECK (calculator), using a tolerance of ±0.01 (±1.00 for grand total after round-off):
   - per item: taxable value x GST% / 100 = tax amount
   - per item: taxable value + tax = line total
   - sum of taxable values = total taxable value
   - sum of tax amounts = total tax
   - total taxable + total tax + round off = grand total
   - amount in words matches grand total

C) DATE CHECK (use TODAY'S DATE given in the user message):
   - invoice date must be a real calendar date (e.g. 2026-02-30 is invalid)
   - invoice date must NOT be after today's date (today itself is acceptable)
   - due date must be on/after the invoice date

D) SUPPLY TYPE CHECK (check_supply_type): IGST for inter-state, CGST+SGST for intra-state.

overall_status: VALID if everything passes, INVALID if any FAIL, NEEDS_REVIEW if only WARNINGs
(e.g. web sources disagree). Keep the summary short and list the problems found."""


# =============================================================================
# 4. AGENTS (with memory)
# =============================================================================

extractor_agent = create_agent(
    model=MODEL,
    tools=[],                      # no tools: invoice text is passed in the prompt
    system_prompt=EXTRACTOR_PROMPT,
    response_format=InvoiceData,
    checkpointer=InMemorySaver(),
    name="invoice_extractor",
)

validator_agent = create_agent(
    model=MODEL,
    tools=[tavily_search, calculator],
    system_prompt=VALIDATOR_PROMPT,
    response_format=ValidationReport,
    checkpointer=InMemorySaver(),
    name="gst_validator",
)



# Initialize the loader with your PDF file path
loader = PDFPlumberLoader("GST_Tax_Invoice.pdf")

# Load the documents/pages
docs = loader.load()
page_content = docs[0].page_content


extracted_invoice = extractor_agent.invoke(
        {"messages": [{"role": "user",
                       "content": "Extract this invoice:\n\n" + page_content}]},
        config={"configurable": {"thread_id": "1001"}},
    )

print("Extracted Invoice Data:")
print(extracted_invoice["structured_response"].model_dump_json(indent=2))
  


final_result = validator_agent.invoke(
        {"messages": [{"role": "user",
                       "content": f"TODAY'S DATE: {date.today().isoformat()}\n\n"
                                  "Validate this invoice:\n" + extracted_invoice["structured_response"].model_dump_json(indent=2)}]},
        config={"configurable": {"thread_id": "1001"}},
    )


print("Final Validation Result:")
print(final_result["structured_response"].model_dump_json(indent=2))
