
def create_drug_repurposing_prompt(disease_name: str, post_text: str, disease_aliases: list[str] | None = None) -> str:
    """
    Generalized version of scleroderma_repurposing_prompt_refined.py's
    create_scleroderma_repurposing_prompt(), parameterized by disease so it
    can be reused across all 20 scraped rare disease communities instead of
    just scleroderma.

    Every disease-specific hardcoded example from the original was replaced:
    the original's drug examples (CellCept, Ofev, nifedipine), lab examples
    (ANA, Scl-70, anti-centromere -- literally the systemic sclerosis
    antibody panel), and symptom examples (Raynaud's, skin tightening) are
    all meaningless or actively misleading when analyzing, say, Long QT
    Syndrome or Microtia posts. Illustrative examples below are either
    generic enough to apply broadly (fatigue, prednisone, CT scan) or
    parameterized on disease_name, rather than hardcoded to one disease.

    Args:
        disease_name: e.g. "narcolepsy", "myasthenia gravis". Used verbatim
            in the prompt, so pass the same disease_name used elsewhere in
            your pipeline (e.g. the "disease_name" field from
            scrape_rare_disease_subreddits.COMMUNITIES) for consistency.
        post_text: the post's full_text (title + selftext + comments).
        disease_aliases: optional synonyms/acronyms (e.g. ["MG"] for
            myasthenia gravis) to help the model recognize the disease under
            alternate names it may see in informal text.
    """
    alias_note = f' (also referred to as {", ".join(disease_aliases)})' if disease_aliases else ""

    prompt_template = f"""You are a biomedical text extraction expert for patient and caregiver social-media posts about {disease_name}{alias_note}.

The text may be informal, misspelled, conversational, emotionally expressive, and may combine the original post with comments from other users.

Your task:
Extract drug/treatment mentions and clinically relevant biological or medical terms that may help a {disease_name} drug repurposing project.

Core rules:
1. Preserve the author's wording in extracted mentions and quotes.
2. Do not infer clinical facts that are not stated.
3. Normalize names only when there is a clear common standard name or generic name.
4. If unsure, use "unclear" or repeat the original mention rather than guessing.
5. Separate drug-level extraction from general medical term extraction.
6. A quote must be verbatim from the text and must be 30 words or fewer.
7. If a category has no relevant information, use [] or null exactly as shown in the schema.
8. Return only valid JSON. Do not include markdown or explanations.

Social-Media Text:
{{post_text}}

Return JSON using this schema:

{{
  "drug_mentions": [
    {{
      "drug_mention": "",
      "normalized_name": "",
      "drug_or_treatment_class": "",
      "condition_treated": "",
      "label_status": "unclear",
      "mention_type": "",
      "progression_after_treatment": "not_reported",
      "final_outcome": "not_reported",
      "adverse_effects": [],
      "dose_or_schedule": "",
      "quote": ""
    }}
  ],
  "medical_terms": {{
    "diseases_conditions": [],
    "symptoms_manifestations": [],
    "labs_biomarkers_tests": [],
    "procedures_monitoring": [],
    "disease_course_or_progression": [],
    "treatment_response_terms": [],
    "adverse_effect_terms": [],
    "care_context": [],
    "lifestyle_or_complementary_interventions": []
  }}
}}

Drug mention extraction:
Extract one object for EACH medication, drug, biologic, supplement, procedure-like therapy, or treatment modality mentioned.
Examples include prescription medications, biologics, steroids, supplements, over-the-counter drugs, infusions or injections, surgical or procedure-based treatments, and unnamed "meds" when no specific drug is given.

Drug fields:

1. drug_mention
Definition: The drug/treatment name exactly as written in the text.
Examples: "prednisone", "MTX", "the new infusion", "PT".

2. normalized_name
Definition: Best generic or standard name.
Examples:
- "CellCept" -> "mycophenolate mofetil"
- "MTX" -> "methotrexate"
- "Plaquenil" -> "hydroxychloroquine"
If unsure, repeat drug_mention.

3. drug_or_treatment_class
Definition: Broad treatment class if stated or obvious from the drug name.
Examples: "immunosuppressant", "biologic", "steroid", "pain medication", "supplement", "physical therapy".
Use "unclear" if uncertain.

4. condition_treated
Definition: Symptom, disease manifestation, or condition the treatment is being used or discussed for, exactly as described.
Examples: "fatigue", "joint pain", "anxiety", "GERD", specific {disease_name} symptoms as described by the author.
Use "unclear" if not stated.

5. label_status
Allowed values: "on_label", "off_label", "unclear".
Rules:
- Default to "unclear".
- Use "off_label" only if the text explicitly says off-label or clearly describes use outside the usual stated context.
- Use "on_label" only if the text itself gives a basis.
- Do not use outside knowledge alone to decide label status.
- Off-label status is about regulatory approval for the condition being discussed, not about how casually, informally, or recreationally the person describes obtaining or taking the drug. A drug that is genuinely approved for the condition is still on_label even if the account describes non-prescribed, borrowed, or recreational use -- informality alone is not evidence of off-label status.

6. mention_type
Allowed values:
- "patient_reports_taking": author/commenter says they are taking, took, tried, started, stopped, or were prescribed it.
- "commenter_suggests": someone recommends, suggests, or mentions another person using it.
- "asking_about": author/commenter asks whether to try it or asks about experience, with no outcome.
- "doctor_recommended_or_prescribed": clinician recommended or prescribed it.
- "research_or_trial_mention": mentioned as part of a study, trial, paper, or research discussion.
- "incidental": mentioned but not clearly related to managing {disease_name} or its symptoms.
- "unclear": not enough context.

7. progression_after_treatment
Allowed values: "helped_or_improved", "did_not_help", "worsened", "stable_or_no_change", "mixed", "unclear", "not_reported".
Definition: The near-term or direct response after taking, starting, stopping, or using the drug/treatment. Focus on whether the author says the medicine or treatment helped the target symptom/disease manifestation after use.
Rules:
- Base this only on a stated treatment-response link.
- Use "helped_or_improved" for phrases such as "helped", "worked", "improved", "relief", or "symptoms got better on it".
- Use "did_not_help" for phrases such as "didn't help", "did not work", "no benefit", or "failed".
- Use "worsened" when the treatment is described as making symptoms or disease worse.
- Use "stable_or_no_change" for phrases such as "no progression on", "kept me stable", or "no change".
- Use "mixed" when the treatment helped some symptoms but not others, or helped but caused limiting problems.
- Use "unclear" when there is vague response language but the direction is not clear.
- Use "not_reported" when the drug is mentioned without any treatment response.
Examples:
- "prednisone helped my joint pain" -> helped_or_improved.
- "the infusion did not help me" -> did_not_help.
- "that supplement made my symptoms worse" -> worsened.
- "no progression on my meds" -> stable_or_no_change.

8. final_outcome
Allowed values: "improved", "worsened_or_progressed", "stable", "relapsed_after_initial_improvement", "mixed", "unclear", "not_reported".
Definition: The later or overall outcome/course after the initial treatment response, when the text gives a longer-term trajectory. This field captures cases where the initial response and later course differ.
Rules:
- Use this only when the text gives an overall or later outcome, not just an immediate response.
- Use "relapsed_after_initial_improvement" when the text says the person improved after treatment but later worsened, flared, relapsed, or progressed again.
- Use "improved" when the longer-term or final state is improved.
- Use "worsened_or_progressed" when the later or final state is worse/progressive.
- Use "stable" when the later or final state is stable or no progression.
- Use "mixed" when the final course includes both durable improvement and persistent/worsened domains.
- Use "unclear" when timing or final state is ambiguous.
- Use "not_reported" when only immediate response is stated or no outcome is stated.
Examples:
- "The treatment stopped the progression. Over the next 10 months I felt much better." -> improved.
- "I improved at first, but later symptoms came back" -> relapsed_after_initial_improvement.
- "stable on this medication for years" -> stable.
- "It helped the pain, but my other symptoms kept progressing" -> mixed.

9. adverse_effects
Definition: Side effects, intolerance, reactions, complications, or reasons for stopping the treatment.
Examples: ["nausea"], ["headache"], ["infection"], ["couldn't tolerate it"].
Use [] if none stated.

10. dose_or_schedule
Definition: Dose, frequency, route, or duration if stated.
Examples: "500 mg twice daily", "monthly infusions", "for 3 months".
Use "" if not stated.

11. quote
Definition: A short verbatim quote, 30 words or fewer, supporting the extraction.

Medical term categories:

1. diseases_conditions
Extract formal and informal disease/condition names, including {disease_name} itself and any related or comorbid conditions mentioned.
Examples: "{disease_name}", plus any comorbid or related conditions the author names.

2. symptoms_manifestations
Extract symptoms, patient-described manifestations, and informal symptom phrases.
Examples: "fatigue", "joint pain", "trouble sleeping", "shortness of breath", "brain fog".

3. labs_biomarkers_tests
Extract labs, antibodies, biomarkers, and measured values.
Examples: "CBC", "genetic testing", "biopsy result", "antibody panel", or any specific test named in the text.

4. procedures_monitoring
Extract diagnostic procedures, imaging, monitoring, and specialist evaluations.
Examples: "CT scan", "MRI", "sleep study", "echo", "specialist appointment".

5. disease_course_or_progression
Extract progression, stability, flare, worsening, improvement, remission, diagnosis timing, or waiting/monitoring language.
Examples: "getting worse", "stable", "not progressing", "flare", "diagnosed last year", "waiting game".

6. treatment_response_terms
Extract response words linked to a treatment or intervention.
Examples: "helped", "worked", "improved", "no benefit", "didn't help", "made it worse", "mixed".

7. adverse_effect_terms
Extract side effect or intolerance language whether or not a specific drug is clear.
Examples: "side effects", "reaction", "couldn't tolerate", "made me sick".

8. care_context
Extract healthcare context relevant to interpretation.
Examples: "specialist", "doctor", "diagnosed", "clinical trial", "research", "prescribed".

9. lifestyle_or_complementary_interventions
Extract non-drug or complementary interventions.
Examples: "diet", "exercise", "supplements", "physical therapy", "heat/cold therapy", "breathing exercises".

Routing rules:
- Brand/generic drug names belong in drug_mentions.
- Symptoms being treated by a drug should appear in condition_treated and may also appear in medical_terms.symptoms_manifestations.
- A term naming {disease_name} itself, or a variant/subtype of it, may be either a disease/condition or a symptom depending on context; preserve the wording and put it in the most relevant field.
- Diagnostic tests, imaging, and monitoring belong in procedures_monitoring; labs, antibodies, and measured biomarker values belong in labs_biomarkers_tests -- only move a procedure to labs_biomarkers_tests if the text gives a measured result from it.
- "helped my [symptom]" is treatment_response_terms plus progression_after_treatment = "helped_or_improved" if linked to a drug.
- "side effects made me stop" is adverse_effect_terms plus progression_after_treatment = "mixed" or "worsened" if linked to a drug.
- If a text describes both short-term response and later course, fill both progression_after_treatment and final_outcome.
- If a text only says whether the treatment helped, fill progression_after_treatment and set final_outcome to "not_reported".
- If a text only describes later disease course without a clear immediate response, set progression_after_treatment to "not_reported" and fill final_outcome if linked to the treatment.
- Do not decide on-label/off-label from outside knowledge alone.
- Do not infer that a drug treated {disease_name} just because it appears in a {disease_name} forum.

Return only the JSON object."""
    return prompt_template.replace("{post_text}", post_text)
