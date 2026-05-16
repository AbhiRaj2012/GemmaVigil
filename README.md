# 🛡️ GemmaVigil: Enterprise AI Verification Layer

**GemmaVigil** is a dual-agent, cross-platform adversarial testing suite designed to audit and verify the safety, privacy, and factual grounding of commercial Large Language Models (LLMs).

Built as an interactive Red Teaming lab, GemmaVigil pits a **Target Model** (the worker) against a strict **Judge Engine** (the guardrail). It systematically bombards the target with adversarial prompts (PII extraction, medical hallucination traps, policy bypasses) and dynamically grades the responses based on semantic intent.

---

## 🚀 Key Features (Our Edge)

* **⚔️ Dual-Agent Cross-Platform Matrix:** Seamlessly mix architectures. Force a massive cloud model (e.g., `gemini-1.5-pro` or `gpt-4o`) to generate answers, and use a lightning-fast local edge model (e.g., `gemma4:e4b` via Ollama) to audit them—or vice versa.
* **🧠 Intent-Based Semantic Grading:** Our Judge evaluates the *intent* of the response rather than relying on strict dictionary matches. It intelligently distinguishes between a model committing a violation and a model safely repeating a word to refuse a request (eliminating "Refusal Echoing" false failures).
* **🛡️ Enterprise Cloud Hardening:** Built-in **5-Attempt Auto-Retry Loops** ensure that API rate limits (HTTP 429), 500 Internal Server Errors, and silent safety blocks from cloud providers never crash your batch evaluations.
* **💾 Continuous History Logging:** GemmaVigil automatically compiles and appends every test, score, and failure reason into a persistent, Excel-ready `gemmavigil_history_log.csv` the exact millisecond a batch finishes. No manual downloads required.
* **📊 Dynamic UI & State Management:** A highly responsive Streamlit dashboard that freezes during execution to prevent mid-run tampering, and instantly paints a severity-weighted Enterprise Analytics Matrix upon completion.
* **🔒 Local-First Privacy:** Full support for 100% offline execution via Ollama, allowing you to audit highly sensitive PII scenarios without ever sending data to third-party servers.

---

## 🗺️ Future Path (Roadmap)

While the **Bruteforce Lab (Phase 1)** provides rigorous offline auditing, the core architecture is designed to evolve into a live production defense system:

1.  **👁️ The Live Surveillance Proxy (Phase 2):** Deploying the Judge Engine as an ultra-fast middleware proxy. Sitting between users and your commercial API, it intercepts and blocks hallucinations or PII leaks in milliseconds *before* they ever reach the end-user.
2.  **⚖️ Precision Scoring (Benign Benchmarking):** Integrating a dataset of safe, benign queries to measure the "False Refusal Rate" (Precision), ensuring the AI isn't overly cautious and blocking legitimate user requests.
3.  **🖼️ Multimodal Auditing:** Extending the pipeline to evaluate Vision-Language Models (VLMs) by testing them against adversarial images and embedded text traps.
4.  **⚙️ Custom Rubric Builder:** A drag-and-drop UI allowing enterprise compliance teams to write and inject their own custom regulatory rules (e.g., specific HIPAA, GDPR, or internal company policies) without touching code.

---
