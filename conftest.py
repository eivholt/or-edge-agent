from dotenv import load_dotenv

# Load .env with override so project-local values
# (e.g. OPENAI_API_KEY=local-dev-key) take precedence
# over any global shell exports.
load_dotenv(override=True)
