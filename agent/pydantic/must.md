### Defensive Schema Engineering
    - Flexible Context Handling: Use Pydantic’s Optional and Field(default=...) to handle the variable nature of the 4-context framework, specifically for CustomerContext which is only present in 16% of the test cases.  

    - Strict Type Casting: Ensure all numerical data from MerchantContext (e.g., performance metrics) and CategoryContext (e.g., peer_stats) are strictly cast to float or int to prevent logic errors during prompt construction.  

### Output Integrity Guardrails
    Response Schema Enforcement: Define a ComposedMessage model to strictly enforce the judge's required output keys: body, cta, send_as, suppression_key, and rationale.  

    Enum-Based Routing: Use Literal for the send_as field (vera | merchant_on_behalf) to ensure the bot never misattributes its identity.  

### Semantic & Specificity Validation
    Hallucination Checkers: Implement @model_validator methods to cross-reference the generated body against the input contexts.

    Verification: If a price or statistic is mentioned in the message, the validator must verify it exists in the offer_catalog or peer_stats.  

    Voice & Tone Verification: Use regex-based validators to check for "Clinical/Peer" tone in the dentists category and "Retail" tone in salons, flagging any "Promotional" anti-patterns like "AMAZING DEAL!".  

### Multi-Lingual Pattern Matching
    Code-Mix Detection: Add validators to confirm the body matches the language_pref (e.g., "hi-en mix") provided in the context.  

    CTA Positioning: Ensure the cta is extracted or placed at the final sentence of the body to avoid the "Buried CTA" penalty.