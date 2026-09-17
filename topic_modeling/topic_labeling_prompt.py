def create_topic_labeling_prompt(disease_name: str, top_keywords: list[str], example_snippets: list[str]) -> str:
    """
    Builds an LLM prompt for Step 7 (topic labeling), same structure/rigor
    as drug_repurposing_prompt.py's create_drug_repurposing_prompt(): a
    parameterized disease name, a strict JSON schema, allowed-value enums,
    worked examples, and explicit anti-over-inference rules.

    This is a pre-labeling DRAFT, not a final label -- both a plain keyword
    heuristic and this LLM pass are known to sometimes miscategorize niche
    clinical vocabulary (a keyword list missed a drug-trial topic entirely
    because it didn't recognize "takeda, centessa, alkermes" as
    pharma-company names; an LLM could plausibly make similar misses on
    unfamiliar drug names). Review every "clinical_treatment" label before
    using it to select documents for Step 8 extraction.

    Args:
        disease_name: e.g. "narcolepsy". Same value used elsewhere in the
            pipeline (scrape_rare_disease_subreddits.COMMUNITIES).
        top_keywords: the topic's top words from Top2Vec, most-similar first.
        example_snippets: 2-3 short excerpts from real documents assigned to
            this topic (doc_topics.json), giving the model actual context
            beyond bare keywords -- keywords alone can be ambiguous (e.g.
            "swelling, hip, joint" could be a symptom topic or a fitness/
            injury topic; the surrounding text disambiguates it).
    """
    keywords_str = ", ".join(top_keywords)
    snippets_str = "\n---\n".join(s.strip()[:500] for s in example_snippets if s.strip())

    return f"""You are helping a researcher triage topics automatically discovered (via Top2Vec topic modeling) in a Reddit community for {disease_name} patients and caregivers, as part of a rare disease drug-repurposing project.

You will be given a topic's top keywords (most representative first) and a few real example excerpts from posts/comments assigned to that topic. Your job is to propose a short label and a relevance category for this topic -- a draft for a human reviewer, not a final decision.

Core rules:
1. Base your judgment on the keywords and example excerpts given. Do not assume content that isn't shown.
2. If the keywords and examples conflict or are ambiguous, prefer "unclear" over guessing.
3. A topic being about {disease_name}-adjacent content (comorbidities, differential diagnoses, related conditions) still counts as clinical_treatment if it involves symptoms, treatments, testing, or disease management -- being off-topic from {disease_name} specifically is not the same as being off-topic from the project's goals.
4. Non-English text, a cluster of only common function words (the, is, to, that, you...), or other signs the "topic" is a modeling artifact rather than real semantic content should be labeled data_artifact, not off_topic.
5. Return only valid JSON. Do not include markdown or explanations outside the JSON.

Topic keywords (most representative first):
{keywords_str}

Example excerpts from posts assigned to this topic:
{snippets_str}

Return JSON using this schema:

{{
  "label": "",
  "category": "",
  "reasoning": ""
}}

Field definitions:

1. label
Definition: A short (3-8 word) human-readable description of what this topic is actually about.
Examples: "IVIG infusion premedication and side effects", "Orexin agonist drug trials", "Job/career discussion", "Non-English (Portuguese) content cluster".

2. category
Allowed values:
- "clinical_treatment": the topic centers on medications, treatments, symptoms, diagnostic tests/procedures, drug trials, disease management, or medically-relevant comorbidities. This is the category worth feeding into downstream drug-signal extraction.
- "support_community": emotional support, encouragement, general community bonding, or day-to-day life content from patients/caregivers that isn't specifically about symptoms or treatment.
- "off_topic": unrelated to the disease, patients, or caregivers at all (e.g., politics, unrelated hobbies, career discussion, entertainment/media recommendations).
- "data_artifact": the topic isn't real semantic content -- a non-English-language cluster, a cluster of mostly generic function words, or keywords too incoherent to represent any actual subject.
- "unclear": the keywords and examples don't give enough signal to confidently pick another category.

Examples:
- Keywords "mycophenolate, cellcept, mmf, dosage, prescribed" with excerpts discussing starting/stopping a medication -> category "clinical_treatment", label "Mycophenolate mofetil (CellCept) dosing".
- Keywords "erlichia, borrelia, tick, babesia, bartonella" with excerpts about tick-borne illness testing -> category "clinical_treatment" (a comorbid/differential-diagnosis condition, not off-topic), label "Lyme disease / tick-borne coinfection discussion".
- Keywords "congratulations, journey, proud, celebrating, inspiring" -> category "support_community", label "Community encouragement and milestones".
- Keywords "trump, republicans, senate, congress, gop" -> category "off_topic", label "US politics discussion".
- Keywords "que, pasando, sentir, medicamentos, tomando" -> category "data_artifact", label "Non-English (Spanish/Portuguese) content cluster".
- Keywords "is, to, not, that, you, the, it, this, of, as" -> category "data_artifact", label "Generic function-word cluster (no coherent topic)".

3. reasoning
Definition: One sentence explaining the category choice, referencing specific keywords or excerpt content.

Return only the JSON object."""
