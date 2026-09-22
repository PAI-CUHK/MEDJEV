"""Lightweight, auditable guards for medical claims before automatic acceptance."""
import re

def medical_guard(state: str, claim: str, prediction: str):
    s=(state or '').lower(); c=(claim or '').lower(); flags=[]
    temps=[float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*(?:degrees?\s*)?(?:c|°c|celsius)',s)]
    if temps and 'fever' in c and max(temps) < 38.0: flags.append('temperature_below_fever_threshold')
    if re.search(r'\b(19|20)\d{2}\b|\b(previous|prior|history of|historical)\b',s) and re.search(r'\b(current|currently|now|present)\b',c): flags.append('historical_vs_current_conflict')
    if ('no history of' in s or 'denies' in s or 'without' in s) and any(x in c for x in ['has diabetes','has chest pain','has pneumonia']): flags.append('explicit_negation_conflict')
    if 'no microbiology results' in s and ('negative' in c or 'positive' in c): flags.append('missing_result_not_result')
    action='review' if flags and prediction=='supported' else ('revise' if flags else 'accept')
    return {'action':action,'flags':flags}
