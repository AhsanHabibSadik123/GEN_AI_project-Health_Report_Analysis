
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq


def build_extraction_prompt(blood_work_data: str) -> str:
	return f"""
You are a medical data extraction assistant.
From the medical data report below, EXTRACT all the test values and classify each one as HIGH, LOW, or NORMAL based on the reference ranges provided in the report.
Format your response as:
- Test Name: Value | Status (HIGH, LOW, NORMAL) | Reference Range

Medical Data Report:
{blood_work_data}
"""


def build_diet_prompt(extracted_data: str) -> str:
	return f"""
You are a nutritionist specialized in Bangladeshi dietary habits.
Based on the blood work data provided, write:
1. A short health summary in 3 lines explaining the current health condition in simple language.
2. A short Bangladeshi diet plan having only two sections: (1) Foods to eat more of, (2) Foods to avoid.
Do not include any other sections in the diet plan.

Blood work data:
{extracted_data}
"""


def get_llm() -> ChatGroq:
	load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
	return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def main() -> None:
	st.set_page_config(page_title="Health Plan", page_icon="🩺", layout="centered")
	st.title("Health Plan App")
	st.write(
		"Upload or paste a blood test report to get a simple health summary and food routine."
	)

	with st.expander("Tips for best results"):
		st.write(
			"- Include reference ranges with each test.\n"
			"- Keep the report text clean and complete.\n"
			"- This tool is informational only, not medical advice."
		)

	uploaded_file = st.file_uploader("Upload blood test report (.txt)", type=["txt"])
	report_text = ""

	if uploaded_file is not None:
		report_text = uploaded_file.read().decode("utf-8", errors="ignore")

	report_text = st.text_area(
		"Or paste the report here",
		value=report_text,
		height=240,
		placeholder="Paste your blood test report...",
	)

	submit = st.button("Analyze Report", type="primary")

	if submit:
		load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
		if not report_text.strip():
			st.error("Please upload or paste a blood test report.")
			return

		if not os.getenv("GROQ_API_KEY"):
			st.error(
				"Missing GROQ_API_KEY. Add it to your environment or .env file and try again."
			)
			return

		with st.spinner("Analyzing report..."):
			llm = get_llm()
			extraction_prompt = build_extraction_prompt(report_text)
			extracted_data = llm.invoke(extraction_prompt)
			diet_prompt = build_diet_prompt(extracted_data.content)
			diet_plan = llm.invoke(diet_prompt)

		st.subheader("Health Summary")
		summary_text, food_text = split_summary_and_food(diet_plan.content)

		if summary_text:
			st.write(summary_text)
		else:
			st.write(diet_plan.content)

		if food_text:
			st.subheader("Food Routine")
			st.write(food_text)


def split_summary_and_food(text: str) -> tuple[str, str]:
	"""Split the model response into a summary and food routine if possible."""
	lower_text = text.lower()
	markers = ["foods to eat", "foods to avoid", "food routine", "diet plan"]

	first_marker_index = None
	for marker in markers:
		idx = lower_text.find(marker)
		if idx != -1:
			first_marker_index = idx
			break

	if first_marker_index is None:
		return text.strip(), ""

	summary = text[:first_marker_index].strip()
	food = text[first_marker_index:].strip()
	return summary, food


if __name__ == "__main__":
	main()

