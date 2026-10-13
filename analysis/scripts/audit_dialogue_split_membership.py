"""CPU audit of recovered split membership against official dialogue metadata."""
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    official = {}
    source_hashes = {}
    for split in ('train', 'dev', 'test'):
        path = ROOT / f'data/retrain_inputs/text/{split}.tsv'
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
        with path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                sid = f"dia{row['Dialogue_id']}_utt{row['Utterance_id']}"
                if sid in official:
                    raise ValueError('Duplicate official sample ID across splits')
                official[sid] = (split, row)
    protocols = {}
    for name, relative in [('original', 'data/retrain_inputs/data.json'),
                           ('debiased', 'data/debiased_mintrec2/data.json')]:
        path = ROOT / relative
        data = json.loads(path.read_text(encoding='utf-8'))
        source_hashes[relative] = sha(path)
        ids, dialogues, sources = {}, {}, {}
        for split in ('train', 'dev', 'test'):
            ids[split] = set()
            dialogues[split] = set()
            sources[split] = set()
            for row in data[split]:
                sid = row['id']
                if sid in ids[split] or sid not in official:
                    raise ValueError('Duplicate or unknown split ID')
                original_split, meta = official[sid]
                if original_split != split or row['label'] != meta['Label']:
                    raise ValueError('Moved sample or mismatched label')
                ids[split].add(sid)
                dialogues[split].add(meta['Dialogue_id'])
                sources[split].add(meta['Source'])
        comparisons = []
        for a, b in itertools.combinations(('train', 'dev', 'test'), 2):
            shared = sorted(dialogues[a] & dialogues[b], key=int)
            comparisons.append({'splits': [a, b], 'shared_sample_ids': len(ids[a] & ids[b]),
                'shared_dialogue_count': len(shared), 'shared_dialogue_ids': shared,
                'shared_sources': sorted(sources[a] & sources[b])})
        protocols[name] = {'counts': {s: {'samples': len(ids[s]), 'dialogues': len(dialogues[s]),
                                           'sources': sorted(sources[s])} for s in ids},
                           'pairwise_overlap': comparisons}
    by_dialogue = defaultdict(set)
    for _, meta in official.values():
        by_dialogue[meta['Dialogue_id']].add(meta['Source'])
    multi_source = {k: sorted(v) for k, v in by_dialogue.items() if len(v) > 1}
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete_metadata_membership_audit', 'input_sha256': source_hashes,
        'analysis_source_sha256': sha(Path(__file__)), 'protocols': protocols,
        'dialogue_ids_with_multiple_sources': multi_source,
        'scope': 'IDs/labels/split placement and official Dialogue_id/Source metadata only. Source overlap does not establish duplicate media or label leakage; no claim of speaker-disjoint or unseen-source evaluation.',
        'training_or_generation_started': False}
    (ROOT/'results/second_review/dialogue_split_membership_audit.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'overlaps': {
        p: [{k: v for k, v in pair.items() if k != 'shared_dialogue_ids'} for pair in r['pairwise_overlap']]
        for p, r in protocols.items()}, 'multi_source_dialogue_ids': len(multi_source)}))


if __name__ == '__main__':
    main()
