# Pinned OpenAI API schema

- Source: `openai/openai-openapi`
- Commit: `4bb21ba8e9213c3d955b69dc3f76dd7537439828` (2026-09-13)
- File: `openapi.json`
- OpenAI spec version: `2.3.0`
- OpenAPI version: `3.1.0`
- Size: `4,152,521` bytes
- License: MIT, upstream notice in `LICENSE` beside this file
- SHA-256: `3d6223349eadfd937624b9e6b8abf596ec2f680a1a367889cf6a6f924e568127`

The validator refuses to run if the bytes do not match this hash. The
Chat Completions parameter ledger derives from this same snapshot.

The snapshot uses both JSON Schema null unions and the older OpenAPI
`nullable: true` keyword. `schema.py` normalizes only that keyword before
validation because the published examples and descriptions explicitly permit
the corresponding null values.
