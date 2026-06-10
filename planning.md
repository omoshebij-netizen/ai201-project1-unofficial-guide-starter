# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
Towson's official dining website does not tell users anything about the unbearable line, which dining hall is actually worth walking to , how to strecth dining dollars to last a semester or what options exisit for students with halal requirements and other things. This is valuable becasue it makes it so much easier for freshamn to navigate their dining options, hellping them adjust to college.
---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | `unimart_reviews.txt` | Student reviews (Rate My Dining-style) | towson.edu dining page + student reviews |
| 2 | `newell_dining_reviews.txt` | Student reviews | towson.edu + student reviews |
| 3 | `reddit_dining_thread.txt` | Reddit thread | r/TowsonUniversity |
| 4 | `west_village_commons_reviews.txt` | Student reviews | towson.edu + student reviews |
| 5 | `tiger_den_union_review.txt` | Student blog | towsondiningsurvivor.wordpress.com |
| 6 | `meal_plan_faq.txt` | Student-written FAQ | Student Government FAQ site |
| 7 | `the_study_library_cafe.txt` | Student reviews | towson.edu + student reviews |
| 8 | `dining_survival_guide.txt` | Student survival guide | Unofficial guide website |
| 9 | `dietary_restrictions_guide.txt` | Community guide | Unofficial guide website |
| 10 | `discord_dining_tips.txt` | Discord |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 400 characters 

**Overlap:** 80 characters 

**Reasoning:** The documents are a mis of short reviews, FAQ and discord/redit threads. 80 characters ensures that sentences split at a chunk boundary preventing retrieval failures. 

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** 'all-MiniLM-L6-v2' via 'sentence-transformers'

**Top-k:** 5 chunks per query

**Production tradeoff reflection:** OpenAI has higher accuracy, especially for nuanced semantic mathces, but costs per embedding and adds dependency. Cohere multilingual embeddings are important if serving a campus a significant internaltional student population. MiniLM has a 256 token limit for long documents this would need aggressive chunking. Local models like MiniLM have zero marginal cost and full data privacy, while API models offer better accuracy but add latency and cost.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What is the best time to go to Newell Hall to avoid crowds? | Before 11:30am or after 2pm on weekdays. The peak crowding is 12pm-1:30pm. |
| 2 | Which dining location has the best coffee on campus? | The Study in Cook Library — serves Jittery Joe's coffee, shortest lines on campus, accepts Dining Dollars. Open Mon-Thu 8am-6pm, Fri 8am-3pm, closed weekends. |
| 3 | Can I use my meal swipe at the Tiger Den Chick-fil-A? | No. Tiger Den (including Chick-fil-A) is retail only — accepts Dining Dollars and Tiger Bucks, NOT meal swipes. Swipes only work at Newell Hall and West Village Commons. |
| 4 | What are the best dining options for students with celiac disease? | West Village Commons is the recommended option — it has a dedicated gluten-free station with strict protocols. Multiple students with celiac have reported it's safer than Newell, where cross-contamination risk is higher. |
| 5 | What can I eat on campus after 9pm on a weeknight? | Unimart is open until 11pm on weekdays. After 11pm, no on-campus options exist — students rely on DoorDash/Uber Eats (Chipotle, Raising Cane's, Jersey Mike's all deliver until midnight or later). |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. A student might ask "Is the food court good?" while documents use the name "Tiger Den." The embedding model should handle this via semantic similarity, but highly colloquial or slang terms may not map cleanly to the document terms. Mitigation: ensure documents include both official names and common nicknames.

2. Some questions require pulling from multiple documents (e.g., "Compare Newell and West Village"). A single retrieved chunk won't answer this — the LLM needs 2-3 relevant chunks from different documents. If top-k is too low, one dining hall's information may dominate. Mitigation: test these multi-location queries specifically in evaluation.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->
Documents (12 .txt files)
        |
        v
[INGESTION] ingest.py
  - Load raw text from docs/ folder
  - Clean: remove metadata headers, normalize whitespace
  - Output: list of (text, source_filename) tuples
        |
        v
[CHUNKING] ingest.py
  - RecursiveCharacterTextSplitter
  - chunk_size=400, overlap=80
  - Output: list of (chunk_text, source, chunk_index)
        |
        v
[EMBEDDING + VECTOR STORE] embed.py
  - Embedding model: all-MiniLM-L6-v2 (sentence-transformers, local)
  - Vector store: ChromaDB (persistent, local)
  - Metadata stored: source filename, chunk index
        |
        v
[RETRIEVAL] query.py
  - Input: user query string
  - Embed query with same model
  - ChromaDB similarity search, top-k=5
  - Output: list of (chunk_text, source, distance)
        |
        v
[GENERATION] query.py
  - LLM: Groq llama-3.3-70b-versatile
  - Prompt: retrieved chunks as context, grounding instruction
  - Output: answer string + list of source documents
        |
        v
[INTERFACE] app.py
  - Gradio web UI
  - Input: text box
  - Output: answer + sources panel
```

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
