# Sample_AI_Agent_APP

# Rental Coordinator AI Agent

This project is an AI-powered rental inquiry assistant for forklift hire businesses. It guides clients through the quoting process by asking some questions and matching the request to the most suitable forklift based on lifting specs, site conditions, and availability.

---

## What It Does

- Accepts natural language rental inquiries (e.g. “Chasing a 2.5T forklift”)
- Asks spec questions to find out users need:
   1) Maximum lifting weight
   2) Maximum load size
   3) Indoor, outdoor, or both
   4) Ground conditions
   5) Turning space or access limits
   6) Rental duration? 2 month
   7) Delivery location
- Matches requirements to the closest forklift(s) in the spec sheet
- Returns a rental quote including:
  - Recommended forklift(s) with specs
  - Rental rates (based on pricing sheet)
  - Delivery note

- This app has been deployed as a web app using **Streamlit Cloud**
- The link to the app: https://sample-ai-agent-app-rental-use-case.streamlit.app/

---

## Requirements 

- **Python 3.10+**
- **Streamlit** for the frontend
- **LangChain** with OpenAI & Pinecone for conversational flow and vector retrieval (optional)
- **Pandas/Numpy** for data processing
- **Unstructured / PDF parsing** for brochure intelligence
- **Tesseract + pdf2image** for OCR if needed

---

Install dependencies using:

```bash
pip install -r requirements.txt


