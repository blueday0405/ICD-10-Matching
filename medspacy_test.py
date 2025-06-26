import medspacy
from medspacy.target_matcher import TargetRule
from medspacy.visualization import visualize_ent

nlp = medspacy.load()


text = "I have a headache and a sore thorat"

doc = nlp(text)

for token in doc:
    print(token.text, token.ent_type_)

nlp.remove_pipe("medspacy_pyrush")
nlp.remove_pipe("medspacy_context")

target_matcher = nlp.get_pipe("medspacy_target_matcher")

dx_text = "Pt is a 63M w/ h/o metastatic carcinoid tumor, HTN and hyperlipidemia"
rules = [
    TargetRule("HTN", "DIAGNOSIS"),
    TargetRule("hyperlipidemia", "DIAGNOSIS"),
]

target_matcher.add(rules)

doc_dx = nlp(dx_text)
visualize_ent(doc_dx)