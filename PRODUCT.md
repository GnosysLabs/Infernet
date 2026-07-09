# Product

## Register

product

## Users

Infernet is for everyday AI users who want a familiar conversational experience without needing to understand peer-to-peer infrastructure or model formats. They use a small, curated catalog of official Infernet models; they never import GGUF files or configure upstream repositories. Technical users can inspect network and runtime details when they intentionally opt into them.

## Product Purpose

Infernet makes community-owned and local AI compute feel like a normal chat product. The network should quietly install official signed packages, discover compute, and route inference while the user remains focused on the conversation. Launch begins with one flagship model, Infernet Chat, based on Gemma 4 26B A4B Instruct QAT Q4_0. Success means the primary workflow feels calm and immediate, while operational information remains understandable and available without dominating the interface.

## Brand Personality

Calm, warm, quietly capable. Infernet should feel trustworthy and human like the best conversational AI products, with technical depth revealed progressively rather than performed visually.

## Anti-references

- Infrastructure dashboards placed inside the conversation.
- Raw shard, peer, checksum, protocol, or layer terminology shown to everyday users.
- Dense grids of metrics whose meaning or next action is unclear.
- Decorative futuristic networking visuals that make a simple chat workflow feel complicated.
- Placeholder assistant messages that explain the product instead of letting the interface teach itself.
- User-facing GGUF importers, repository fields, access tokens, or arbitrary model compatibility claims.

## Design Principles

1. Conversation first. Chat content and the composer own the primary surface.
2. Hide the machinery, preserve trust. Explain outcomes in plain language and reveal technical details only on request.
3. Show progress that answers user questions: what is happening, how long it has taken, and whether action is needed.
4. Use progressive disclosure. Activity belongs in a collapsible secondary surface, with developer details reserved for Developer Mode.
5. Make every state actionable. Empty, loading, download, success, and error states should tell users what matters next.
6. Curate the runtime. Infernet controls, tests, signs, and distributes every model package exposed in the product.

## Accessibility & Inclusion

Target WCAG AA contrast and interaction patterns. All controls must be keyboard accessible with visible focus states and meaningful labels. Respect reduced-motion preferences, avoid communicating state with color alone, and keep status updates understandable to assistive technology.
