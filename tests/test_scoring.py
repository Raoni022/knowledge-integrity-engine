from src.kie.profile_delta import profile_delta_score
from src.kie.scoring import consistency_score, genericity_score, scar_score
from src.kie.web_signals import web_signals_score



def test_scar_score_rewards_operational_detail():
    text = "We had a pgvector lock, tuned top-k, and added retry logic after webhook dedup failures."
    assert scar_score(text) > 0.5



def test_genericity_score_catches_vague_language():
    text = "It is important to follow best practices for scalable and robust systems."
    assert genericity_score(text) > 0.4



def test_consistency_score_rewards_tool_plus_context():
    tools = ["LangChain", "Docker", "pgvector"]
    text = "In LangChain we changed chunking, then a pgvector timeout forced a rollback in Docker."
    assert consistency_score(tools, text, 5) > 0.45



def test_profile_delta_flags_unusual_growth():
    score = profile_delta_score({"timeline_anomaly": 0.9, "title_inflation": 0.8, "years_experience": 2})
    assert score > 0.75



def test_web_signal_score_is_high_when_public_evidence_is_weak():
    score = web_signals_score({"github_activity": 0.1, "public_presence": 0.1, "temporal_coherence": 0.2})
    assert score > 0.7
